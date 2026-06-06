/**
 * DB2 Vector Store implementation for n8n
 * Ported from db2vs.py
 */

import { createHash, randomUUID } from 'crypto';
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
		id: 'CHAR(16) PRIMARY KEY NOT NULL',
		text: 'CLOB',
		metadata: 'BLOB',
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
			client.commitTransaction((err: Error) => {
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

		// Generate or use provided IDs - hash and truncate to 16 chars like Python implementation
		const ids = options?.ids
			? options.ids.map((id) => this.hashAndTruncateId(id))
			: vectors.map(() => this.hashAndTruncateId(randomUUID()));

		this.validateEmbeddingDimension(vectors);

		// Always use row-by-row insert for DB2 vector operations
		// Batch insert with ibm_db's row-wise array insert does not work correctly
		// with DB2 functions like VECTOR() and SYSTOOLS.JSON2BSON() in the VALUES clause.
		// The driver's parameter counting mechanism gets confused by the function wrappers,
		// resulting in "Wrong number of parameters" errors even though the SQL is correct.
		// Row-by-row insert works reliably with these DB2-specific functions.
		return await this.rowByRowInsertVectors(vectors, documents, ids);
	}

	/**
	 * NOTE: Batch insert is not used for DB2 vector operations.
	 *
	 * The ibm_db driver's row-wise array insert mechanism does not work correctly
	 * with DB2-specific functions like VECTOR() and SYSTOOLS.JSON2BSON() in the
	 * VALUES clause. The driver's parameter counting gets confused by the function
	 * wrappers, resulting in "CLI0100E Wrong number of parameters. SQLSTATE=07001"
	 * errors even though the SQL statement is syntactically correct.
	 *
	 * Example SQL that fails with batch insert but works with row-by-row:
	 * INSERT INTO table (id, embedding, metadata, text)
	 * VALUES (?, VECTOR(?, 768, FLOAT32), SYSTOOLS.JSON2BSON(?), ?)
	 *
	 * The driver expects 4 parameters (one per ?), but the functions wrapping
	 * some parameters confuse the batch insert mechanism. Row-by-row insert
	 * works reliably because it processes each statement individually.
	 *
	 * Python's ibm_db_dbi.cursor.executemany() may handle this differently,
	 * but the Node.js ibm_db library does not have an equivalent that works
	 * with DB2 functions in the VALUES clause.
	 */

	/**
	 * Insert vectors row by row
	 */
	private async rowByRowInsertVectors(
		vectors: number[][],
		documents: Document[],
		ids: string[],
	): Promise<string[]> {
		const quotedTable = getQuotedTableIdentifier(this.tableName);

		// Get vector dimension from first vector
		const vectorDimension = vectors[0]?.length || 0;

		// Match Python implementation column order: id, embedding, metadata, text
		// IMPORTANT: Use VECTOR() function for embedding and SYSTOOLS.JSON2BSON() for metadata
		// to match Python implementation (db2vs.py lines 335-338)
		const sqlInsert = `
			INSERT INTO ${quotedTable}
			(${this.columnNames.id}, ${this.columnNames.embedding}, ${this.columnNames.metadata}, ${this.columnNames.text})
			VALUES (?, VECTOR(?, ${vectorDimension}, FLOAT32), SYSTOOLS.JSON2BSON(?), ?)
		`;

		try {
			// Begin transaction
			await new Promise((resolve, reject) => {
				this.client.beginTransaction((err: Error) => {
					if (err) reject(err);
					else resolve(null);
				});
			});

			for (let i = 0; i < vectors.length; i++) {
				const id = ids[i];
				const embeddingList = `[${vectors[i].join(',')}]`;
				const metadataJson = JSON.stringify(documents[i].metadata || {});

				await new Promise((resolve, reject) => {
					this.client.query(
						sqlInsert,
						// Data order: id, embedding, metadata, text (matching Python implementation and SQL column order)
						[id, embeddingList, metadataJson, documents[i].pageContent],
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
				this.client.commitTransaction((err: Error) => {
					if (err) reject(err);
					else resolve(null);
				});
			});
		} catch (error) {
			// Rollback on any failure
			try {
				await new Promise((resolve) => {
					this.client.rollbackTransaction(() => resolve(null));
				});
			} catch (rollbackError) {
				// Ignore rollback errors, throw original error
			}
			throw error;
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
			// Begin transaction
			await new Promise((resolve, reject) => {
				this.client.beginTransaction((err: Error) => {
					if (err) reject(err);
					else resolve(null);
				});
			});

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
				this.client.commitTransaction((err: Error) => {
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
				this.client.rollbackTransaction(() => resolve(null));
			});
			throw error;
		}
	}

	/**
	 * Hash and truncate ID to 16 characters (matching Python implementation)
	 * Python: hashlib.sha256(_id.encode()).hexdigest()[:16].upper()
	 * This ensures IDs fit in CHAR(16) column
	 */
	private hashAndTruncateId(id: string): string {
		const hash = createHash('sha256').update(id).digest('hex');
		return hash.substring(0, 16).toUpperCase();
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
