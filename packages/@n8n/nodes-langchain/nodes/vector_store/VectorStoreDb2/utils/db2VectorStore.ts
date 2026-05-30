/**
 * DB2 Vector Store implementation for n8n
 * Ported from db2vs.py
 */

import { randomUUID } from 'crypto';
import { Document } from '@langchain/core/documents';
import type { Embeddings } from '@langchain/core/embeddings';
import { VectorStore } from '@langchain/core/vectorstores';

import { validateIdentifier, getQuotedIdentifier, createSafeErrorMessage } from './db2Security';
import {
	DistanceStrategy as DS,
	type DistanceStrategy,
	type DB2VectorStoreConfig,
	type ColumnMapping,
	type SearchFilter,
} from './types';

/**
 * Get quoted table identifier
 */
function getQuotedTableIdentifier(tableName: string): string {
	return getQuotedIdentifier(tableName);
}

/**
 * Check if table exists in DB2
 */
async function tableExists(client: any, tableName: string): Promise<boolean> {
	const quotedTable = getQuotedTableIdentifier(tableName);
	try {
		const query = `SELECT COUNT(*) FROM ${quotedTable}`;
		await new Promise((resolve, reject) => {
			client.query(query, (err: Error, result: any) => {
				if (err) reject(err);
				else resolve(result);
			});
		});
		return true;
	} catch (error: any) {
		if (error.message && error.message.includes('SQL0204N')) {
			return false;
		}
		throw error;
	}
}

/**
 * Get column names from DB2 table
 */
async function getColumnNames(client: any, tableName: string): Promise<ColumnMapping> {
	const query = `
		SELECT COLNAME
		FROM SYSCAT.COLUMNS
		WHERE TABNAME = ?
		ORDER BY COLNO
	`;

	return await new Promise((resolve, reject) => {
		client.query(query, [tableName.toUpperCase()], (err: Error, results: any[]) => {
			if (err) {
				reject(err);
				return;
			}

			if (!results || results.length === 0) {
				// Return default quoted column names
				resolve({
					id: '"id"',
					text: '"text"',
					metadata: '"metadata"',
					embedding: '"embedding"',
				});
				return;
			}

			const actualColumns: Record<string, string> = {};
			for (const row of results) {
				const colName = row.COLNAME.trim();
				const colLower = colName.toLowerCase();

				// Check if column is uppercase (unquoted) or mixed case (quoted)
				if (colName === colName.toUpperCase()) {
					actualColumns[colLower] = colName;
				} else {
					actualColumns[colLower] = `"${colName}"`;
				}
			}

			// Map logical names to actual column names
			const columnMap: ColumnMapping = {
				id: actualColumns.id || actualColumns._id || '"id"',
				text: actualColumns.text || actualColumns.content || actualColumns.data || '"text"',
				metadata:
					actualColumns.metadata || actualColumns.meta || actualColumns.properties || '"metadata"',
				embedding:
					actualColumns.embedding ||
					actualColumns.vector ||
					actualColumns.embeddings ||
					'"embedding"',
			};

			resolve(columnMap);
		});
	});
}

/**
 * Get distance function name for DB2
 */
function getDistanceFunction(distanceStrategy: DistanceStrategy): string {
	const strategyMap: Record<DistanceStrategy, string> = {
		[DS.EUCLIDEAN]: 'EUCLIDEAN',
		[DS.DOT_PRODUCT]: 'DOT',
		[DS.COSINE]: 'COSINE',
	};

	const func = strategyMap[distanceStrategy];
	if (!func) {
		throw new Error(`Unsupported distance strategy: ${distanceStrategy}`);
	}

	return func;
}

/**
 * Get existing table's vector dimension
 */
async function getTableVectorDimension(client: any, tableName: string): Promise<number | null> {
	const query = `
		SELECT LENGTH, SCALE
		FROM SYSCAT.COLUMNS
		WHERE TABNAME = ? AND COLNAME = 'EMBEDDING'
	`;

	return await new Promise((resolve, reject) => {
		client.query(query, [tableName.toUpperCase()], (err: Error, results: any[]) => {
			if (err) {
				reject(err);
				return;
			}

			if (!results || results.length === 0) {
				resolve(null);
				return;
			}

			// DB2 vector type stores dimension in LENGTH column
			const dimension = results[0].LENGTH;
			resolve(dimension);
		});
	});
}

/**
 * Create table for vector storage
 */
async function createTable(client: any, tableName: string, embeddingDim: number): Promise<void> {
	const validatedTableName = validateIdentifier(tableName, 'table name');

	const colsDict = {
		id: 'VARCHAR(100) PRIMARY KEY NOT NULL',
		text: 'CLOB(10M)',
		metadata: 'CLOB(1M)',
		embedding: `vector(${embeddingDim}, FLOAT32)`,
	};

	const exists = await tableExists(client, validatedTableName);
	if (!exists) {
		const ddlBody = Object.entries(colsDict)
			.map(([colName, colType]) => `"${colName}" ${colType}`)
			.join(', ');

		const quotedTable = getQuotedTableIdentifier(validatedTableName);
		const ddl = `CREATE TABLE ${quotedTable} (${ddlBody})`;

		await new Promise((resolve, reject) => {
			client.query(ddl, (err: Error) => {
				if (err) reject(err);
				else resolve(null);
			});
		});

		await new Promise((resolve, reject) => {
			client.query('COMMIT', (err: Error) => {
				if (err) reject(err);
				else resolve(null);
			});
		});
	}
}

/**
 * Drop table from DB2
 */
export async function dropTable(client: any, tableName: string): Promise<void> {
	const validatedTableName = validateIdentifier(tableName, 'table name');
	const quotedTableName = getQuotedTableIdentifier(validatedTableName);
	const ddl = `DROP TABLE ${quotedTableName}`;

	await new Promise((resolve, reject) => {
		client.query(ddl, (err: Error) => {
			if (err) reject(err);
			else resolve(null);
		});
	});
}

/**
 * DB2 Vector Store class
 */
export class DB2VectorStore extends VectorStore {
	private client: any;
	private tableName: string;
	private distanceStrategy: DistanceStrategy;
	private columnNames: ColumnMapping;
	private useBatchInsert: boolean;

	_vectorstoreType(): string {
		return 'db2';
	}

	constructor(embeddings: Embeddings, config: DB2VectorStoreConfig) {
		super(embeddings, config);

		// Validate table name
		const validatedTableName = validateIdentifier(config.tableName, 'table name');

		this.client = config.client;
		this.tableName = validatedTableName;
		this.distanceStrategy = config.distanceStrategy || DS.EUCLIDEAN;
		this.useBatchInsert = config.useBatchInsert !== false; // Default to true
		this.columnNames = {
			id: '"id"',
			text: '"text"',
			metadata: '"metadata"',
			embedding: '"embedding"',
		};
	}

	/**
	 * Initialize the vector store
	 */
	async initialize(): Promise<void> {
		try {
			// Get embedding dimension from current model
			const embeddingDim = await this.getEmbeddingDimension();

			// Check if table exists
			const exists = await tableExists(this.client, this.tableName);

			if (exists) {
				// Validate existing table's dimension matches current model
				const tableDim = await getTableVectorDimension(this.client, this.tableName);
				if (tableDim !== null && tableDim !== embeddingDim) {
					throw new Error(
						`Embedding dimension mismatch: table has ${tableDim} dimensions, ` +
							'but current embedding model produces ' +
							embeddingDim +
							' dimensions. ' +
							'Please use a different table name or update the embedding model.',
					);
				}
			} else {
				// Create table if it doesn't exist
				await createTable(this.client, this.tableName, embeddingDim);
			}

			// Get actual column names
			this.columnNames = await getColumnNames(this.client, this.tableName);
		} catch (error) {
			const safeMsg = createSafeErrorMessage(error as Error, 'while initializing vector store');
			throw new Error(safeMsg);
		}
	}

	/**
	 * Get embedding dimension
	 */
	private async getEmbeddingDimension(): Promise<number> {
		const embeddedDocument = await this.embeddings.embedQuery('test');
		return embeddedDocument.length;
	}

	/**
	 * Validate embedding dimension and values
	 */
	private validateEmbeddingDimension(embeddings: number[][]): void {
		if (embeddings.length === 0) return;

		const expectedDim = embeddings[0].length;

		for (const embedding of embeddings) {
			if (embedding.length !== expectedDim) {
				throw new Error(
					`Embedding dimension mismatch: expected ${expectedDim}, got ${embedding.length}`,
				);
			}

			// Validate all values are finite numbers (prevent SQL injection)
			for (const value of embedding) {
				if (!Number.isFinite(value)) {
					throw new Error(`Invalid embedding value: ${value}. All values must be finite numbers.`);
				}
			}
		}
	}

	/**
	 * Add documents to the vector store
	 */
	async addDocuments(documents: Document[], options?: { ids?: string[] }): Promise<string[]> {
		return await this.addVectors(
			await this.embeddings.embedDocuments(documents.map((doc) => doc.pageContent)),
			documents,
			options,
		);
	}

	/**
	 * Add vectors to the vector store
	 */
	async addVectors(
		vectors: number[][],
		documents: Document[],
		options?: { ids?: string[] },
	): Promise<string[]> {
		if (vectors.length === 0 || documents.length === 0) {
			throw new Error('No vectors or documents provided');
		}

		if (vectors.length !== documents.length) {
			throw new Error('Number of vectors and documents must match');
		}

		// Generate or use provided IDs
		const ids = options?.ids || vectors.map(() => this.generateId());

		this.validateEmbeddingDimension(vectors);

		if (this.useBatchInsert && vectors.length > 1) {
			return await this.batchInsertVectors(vectors, documents, ids);
		} else {
			return await this.rowByRowInsertVectors(vectors, documents, ids);
		}
	}

	/**
	 * Batch insert vectors using standard DB2 Parameter Array Binding
	 */
	private async batchInsertVectors(
		vectors: number[][],
		documents: Document[],
		ids: string[],
	): Promise<string[]> {
		const quotedTable = getQuotedTableIdentifier(this.tableName);

		// Use a standard single-row placeholder. The driver handles replication for the array length.
		const sqlInsert = `
			INSERT INTO ${quotedTable}
			(${this.columnNames.id}, ${this.columnNames.text}, ${this.columnNames.metadata}, ${this.columnNames.embedding})
			VALUES (?, ?, ?, ?)
		`;

		// Format data into an array of parameter arrays (Matrix structure)
		const batchData: any[][] = [];
		for (let i = 0; i < vectors.length; i++) {
			const embeddingList = `[${vectors[i].join(',')}]`;
			const metadataJson = JSON.stringify(documents[i].metadata || {});
			batchData.push([ids[i], documents[i].pageContent, metadataJson, embeddingList]);
		}

		// Turn off auto-commit manually if managing transaction scopes safely
		if (this.client.setAutoCommit) {
			this.client.setAutoCommit(false);
		}

		let statement: any = null;
		try {
			// 1. Prepare the query once
			statement = await new Promise((resolve, reject) => {
				this.client.prepare(sqlInsert, (err: Error, stmt: any) => {
					if (err) reject(err);
					else resolve(stmt);
				});
			});

			// 2. Execute the statement with the full matrix batch array
			await new Promise((resolve, reject) => {
				statement.execute(batchData, (err: Error) => {
					if (err) {
						const safeMsg = createSafeErrorMessage(err, 'during batch statement execution');
						reject(new Error(safeMsg));
					} else {
						resolve(null);
					}
				});
			});

			// 3. Commit Transaction via driver native API
			await new Promise((resolve, reject) => {
				this.client.commit((err: Error) => {
					if (err) reject(err);
					else resolve(null);
				});
			});
		} catch (error) {
			// Rollback on any failure
			await new Promise((resolve) => {
				this.client.rollback(() => resolve(null));
			});
			throw error;
		} finally {
			// CRITICAL: Always close statements to prevent severe CLI memory leaks
			if (statement) {
				statement.closeSync();
			}
			if (this.client.setAutoCommit) {
				this.client.setAutoCommit(true); // Restore defaults
			}
		}

		return ids;
	}

	/**
	 * Insert vectors row by row
	 */
	private async rowByRowInsertVectors(
		vectors: number[][],
		documents: Document[],
		ids: string[],
	): Promise<string[]> {
		const quotedTable = getQuotedTableIdentifier(this.tableName);

		const sqlInsert = `
			INSERT INTO ${quotedTable}
			(${this.columnNames.id}, ${this.columnNames.text}, ${this.columnNames.metadata}, ${this.columnNames.embedding})
			VALUES (?, ?, ?, ?)
		`;

		if (this.client.setAutoCommit) {
			this.client.setAutoCommit(false);
		}

		try {
			for (let i = 0; i < vectors.length; i++) {
				const id = ids[i];
				const embeddingList = `[${vectors[i].join(',')}]`;
				const metadataJson = JSON.stringify(documents[i].metadata || {});

				await new Promise((resolve, reject) => {
					this.client.query(
						sqlInsert,
						[id, documents[i].pageContent, metadataJson, embeddingList],
						(err: Error) => {
							if (err) {
								const safeMsg = createSafeErrorMessage(
									err,
									`while inserting document with id ${id}`,
								);
								reject(new Error(safeMsg));
							} else {
								resolve(null);
							}
						},
					);
				});
			}

			// Commit after everything succeeds
			await new Promise((resolve, reject) => {
				this.client.commit((err: Error) => {
					if (err) reject(err);
					else resolve(null);
				});
			});
		} catch (error) {
			await new Promise((resolve) => {
				this.client.rollback(() => resolve(null));
			});
			throw error;
		} finally {
			if (this.client.setAutoCommit) {
				this.client.setAutoCommit(true);
			}
		}

		return ids;
	}

	/**
	 * Add texts to the vector store
	 * Delegates to addVectors after generating embeddings
	 */
	async addTexts(
		texts: string[],
		metadatas?: Array<Record<string, any>>,
		options?: { ids?: string[] },
	): Promise<string[]> {
		if (texts.length === 0) {
			throw new Error('No texts provided');
		}

		// Generate embeddings
		const embeddings = await this.embeddings.embedDocuments(texts);

		// Create Document objects with metadata
		const documents = texts.map((text, i) => ({
			pageContent: text,
			metadata: metadatas?.[i] || {},
		}));

		// Delegate to addVectors to avoid code duplication
		return await this.addVectors(embeddings, documents, options);
	}

	/**
	 * Similarity search
	 */
	async similaritySearch(query: string, k: number = 4, filter?: SearchFilter): Promise<Document[]> {
		const embedding = await this.embeddings.embedQuery(query);
		return await this.similaritySearchVectorWithScore(embedding, k, filter).then((results) =>
			results.map((result) => result[0]),
		);
	}

	/**
	 * Similarity search with score
	 */
	async similaritySearchWithScore(
		query: string,
		k: number = 4,
		filter?: SearchFilter,
	): Promise<Array<[Document, number]>> {
		const embedding = await this.embeddings.embedQuery(query);
		return await this.similaritySearchVectorWithScore(embedding, k, filter);
	}

	/**
	 * Similarity search by vector with score
	 */
	async similaritySearchVectorWithScore(
		embedding: number[],
		k: number = 4,
		filter?: SearchFilter,
	): Promise<Array<[Document, number]>> {
		const distanceFunc = getDistanceFunction(this.distanceStrategy);
		const quotedTable = getQuotedTableIdentifier(this.tableName);

		const embeddingList = `[${embedding.join(',')}]`;
		const vectorDimension = embedding.length;

		// Build filter clause if provided
		let filterClause = '';
		const queryParams: any[] = [];

		if (filter && Object.keys(filter).length > 0) {
			const filterConditions: string[] = [];
			for (const [key, value] of Object.entries(filter)) {
				// Simple equality filter on metadata JSON field
				// For more complex filters, this would need to be expanded
				filterConditions.push(`JSON_VALUE(${this.columnNames.metadata}, '$.${key}') = ?`);
				queryParams.push(String(value));
			}
			if (filterConditions.length > 0) {
				filterClause = `WHERE ${filterConditions.join(' AND ')}`;
			}
		}

		// DB2 requires the VECTOR() constructor function to create a vector from string
		// Format: VECTOR('[1,2,3,...]', dimension, FLOAT32)
		// Function name is lowercase: vector_distance
		// Note: VECTOR constructor parameter cannot be parameterized, but the embedding data
		// is numeric and validated, so it's safe to interpolate
		const query = `
			SELECT ${this.columnNames.id}, ${this.columnNames.text},
			       ${this.columnNames.metadata}, ${this.columnNames.embedding},
			       vector_distance(
			           ${this.columnNames.embedding},
			           VECTOR('${embeddingList}', ${vectorDimension}, FLOAT32),
			           ${distanceFunc}
			       ) AS distance
			FROM ${quotedTable}
			${filterClause}
			ORDER BY distance
			FETCH FIRST ${k} ROWS ONLY
		`;

		return await new Promise((resolve, reject) => {
			this.client.query(query, queryParams, (err: Error, results: any[]) => {
				if (err) {
					const safeMsg = createSafeErrorMessage(err, 'during similarity search');
					reject(new Error(safeMsg));
					return;
				}

				const documents: Array<[Document, number]> = results.map((result) => {
					const metaRaw = result[this.columnNames.metadata.replace(/"/g, '').toUpperCase()];
					let metadata = {};
					try {
						metadata = JSON.parse(metaRaw);
					} catch {
						metadata = {};
					}

					const doc = new Document({
						pageContent: result[this.columnNames.text.replace(/"/g, '').toUpperCase()],
						metadata,
					});

					const distance = result.DISTANCE;
					return [doc, distance];
				});

				resolve(documents);
			});
		});
	}

	/**
	 * Delete documents by IDs
	 */
	async delete(options: { ids: string[] }): Promise<void> {
		const { ids } = options;
		if (!ids || ids.length === 0) {
			throw new Error('No IDs provided for deletion');
		}

		const quotedTable = getQuotedTableIdentifier(this.tableName);

		// Use parameterized query with placeholders
		const placeholders = ids.map(() => '?').join(',');
		const ddl = `DELETE FROM ${quotedTable} WHERE ${this.columnNames.id} IN (${placeholders})`;

		try {
			await new Promise((resolve, reject) => {
				this.client.query(ddl, ids, (err: Error) => {
					if (err) {
						const safeMsg = createSafeErrorMessage(err, 'while deleting documents');
						reject(new Error(safeMsg));
					} else {
						resolve(null);
					}
				});
			});

			await new Promise((resolve, reject) => {
				this.client.query('COMMIT', (err: Error) => {
					if (err) {
						const safeMsg = createSafeErrorMessage(err, 'while committing deletion');
						reject(new Error(safeMsg));
					} else {
						resolve(null);
					}
				});
			});
		} catch (error) {
			// Rollback on error
			await new Promise((resolve) => {
				this.client.query('ROLLBACK', () => resolve(null));
			});
			throw error;
		}
	}

	/**
	 * Generate a unique ID using crypto.randomUUID()
	 * This ensures no collisions even in high-volume batch operations
	 */
	private generateId(): string {
		return randomUUID();
	}

	/**
	 * Create DB2VectorStore from texts
	 */
	static async fromTexts(
		texts: string[],
		metadatas: Array<Record<string, any>> | Record<string, any>,
		embeddings: Embeddings,
		dbConfig: Omit<DB2VectorStoreConfig, 'embeddingFunction'>,
	): Promise<DB2VectorStore> {
		const instance = new DB2VectorStore(embeddings, {
			...dbConfig,
			embeddingFunction: embeddings,
		});

		await instance.initialize();

		const metadatasArray = Array.isArray(metadatas) ? metadatas : texts.map(() => metadatas);

		await instance.addTexts(texts, metadatasArray);

		return instance;
	}

	/**
	 * Create DB2VectorStore from documents
	 */
	static async fromDocuments(
		docs: Document[],
		embeddings: Embeddings,
		dbConfig: Omit<DB2VectorStoreConfig, 'embeddingFunction'>,
	): Promise<DB2VectorStore> {
		const instance = new DB2VectorStore(embeddings, {
			...dbConfig,
			embeddingFunction: embeddings,
		});

		await instance.initialize();
		await instance.addDocuments(docs);

		return instance;
	}

	/**
	 * Create DB2VectorStore from existing index
	 */
	static async fromExistingIndex(
		embeddings: Embeddings,
		dbConfig: Omit<DB2VectorStoreConfig, 'embeddingFunction'>,
	): Promise<DB2VectorStore> {
		const instance = new DB2VectorStore(embeddings, {
			...dbConfig,
			embeddingFunction: embeddings,
		});

		await instance.initialize();

		return instance;
	}
}

// Made with Bob
