import { DB2VectorStore } from '../db2VectorStore';
import type { Embeddings } from '@langchain/core/embeddings';
import type { Document } from '@langchain/core/documents';
import { DistanceStrategy } from '../types';

describe('DB2VectorStore', () => {
	let mockClient: any;
	let mockEmbeddings: jest.Mocked<Embeddings>;
	let vectorStore: DB2VectorStore;

	beforeEach(() => {
		// Create mock database client
		mockClient = {
			query: jest.fn((_sql: string, paramsOrCallback: any, callback?: any) => {
				// Handle both query(sql, callback) and query(sql, params, callback)
				const cb = typeof paramsOrCallback === 'function' ? paramsOrCallback : callback;
				if (cb) {
					// Simulate async callback
					setImmediate(() => cb(null, []));
				}
			}),
			prepare: jest.fn(),
			commit: jest.fn((callback: any) => {
				if (callback) setImmediate(() => callback(null));
			}),
			rollback: jest.fn((callback: any) => {
				if (callback) setImmediate(() => callback(null));
			}),
			close: jest.fn(),
		};

		// Create mock embeddings
		mockEmbeddings = {
			embedDocuments: jest.fn(),
			embedQuery: jest.fn(),
		} as any;
	});

	afterEach(() => {
		jest.clearAllMocks();
	});

	describe('constructor', () => {
		it('should create instance with required parameters', () => {
			vectorStore = new DB2VectorStore(mockEmbeddings, {
				client: mockClient,
				embeddingFunction: mockEmbeddings,
				tableName: 'test_vectors',
			});

			expect(vectorStore).toBeInstanceOf(DB2VectorStore);
		});

		it('should accept optional distance strategy', () => {
			vectorStore = new DB2VectorStore(mockEmbeddings, {
				client: mockClient,
				embeddingFunction: mockEmbeddings,
				tableName: 'test_vectors',
				distanceStrategy: DistanceStrategy.COSINE,
			});

			expect(vectorStore).toBeInstanceOf(DB2VectorStore);
		});

		it('should validate table name', () => {
			expect(
				() =>
					new DB2VectorStore(mockEmbeddings, {
						client: mockClient,
						embeddingFunction: mockEmbeddings,
						tableName: '123invalid',
					}),
			).toThrow('table name');
		});
	});

	describe('initialize', () => {
		beforeEach(() => {
			vectorStore = new DB2VectorStore(mockEmbeddings, {
				client: mockClient,
				embeddingFunction: mockEmbeddings,
				tableName: 'test_vectors',
			});
		});

		it('should create table if it does not exist', async () => {
			// Mock embedQuery to return test embedding
			mockEmbeddings.embedQuery.mockResolvedValue([0.1, 0.2, 0.3]);

			// Mock table doesn't exist (query throws SQL0204N error)
			mockClient.query.mockImplementation((sql: string, paramsOrCallback: any, callback?: any) => {
				const cb = typeof paramsOrCallback === 'function' ? paramsOrCallback : callback;
				if (sql.includes('COUNT(*)')) {
					// Table check query - simulate table doesn't exist
					const error: any = new Error('SQL0204N');
					error.message = 'SQL0204N';
					setImmediate(() => cb(error));
				} else {
					// Other queries succeed
					setImmediate(() => cb(null, []));
				}
			});

			await vectorStore.initialize();

			// Should call embedQuery to get dimension
			expect(mockEmbeddings.embedQuery).toHaveBeenCalledWith('test');

			// Should attempt to create table
			expect(mockClient.query).toHaveBeenCalledWith(
				expect.stringContaining('CREATE TABLE'),
				expect.any(Function),
			);
		});

		it('should not create table if it already exists', async () => {
			mockEmbeddings.embedQuery.mockResolvedValue([0.1, 0.2, 0.3]);

			// Mock table exists and column query
			mockClient.query.mockImplementation((sql: string, paramsOrCallback: any, callback?: any) => {
				const cb = typeof paramsOrCallback === 'function' ? paramsOrCallback : callback;

				if (
					sql.includes('SYSCAT.COLUMNS') &&
					sql.includes('TABNAME') &&
					sql.includes('COLNAME') &&
					!sql.includes('LENGTH')
				) {
					// Return column names for getColumnNames
					setImmediate(() =>
						cb(null, [
							{ COLNAME: 'ID' },
							{ COLNAME: 'TEXT' },
							{ COLNAME: 'METADATA' },
							{ COLNAME: 'EMBEDDING' },
						]),
					);
				} else if (
					sql.includes('SYSCAT.COLUMNS') &&
					sql.includes('LENGTH') &&
					sql.includes('EMBEDDING')
				) {
					// Return vector dimension for getTableVectorDimension
					setImmediate(() => cb(null, [{ LENGTH: 3, SCALE: 0 }]));
				} else if (sql.includes('SELECT COUNT')) {
					// Table exists check
					setImmediate(() => cb(null, [{ '1': 1 }]));
				} else {
					// Default: table exists
					setImmediate(() => cb(null, []));
				}
			});

			await vectorStore.initialize();

			// Should not create table
			const createTableCalls = (mockClient.query as jest.Mock).mock.calls.filter((call) =>
				call[0].includes('CREATE TABLE'),
			);
			expect(createTableCalls).toHaveLength(0);
		});
	});

	describe('addVectors', () => {
		beforeEach(async () => {
			vectorStore = new DB2VectorStore(mockEmbeddings, {
				client: mockClient,
				embeddingFunction: mockEmbeddings,
				tableName: 'test_vectors',
				useBatchInsert: false, // Use row-by-row for these tests
			});

			mockEmbeddings.embedQuery.mockResolvedValue([0.1, 0.2, 0.3]);
			await vectorStore.initialize();
			jest.clearAllMocks();

			// Re-create essential mocks after clearing
			const mockStatement = {
				execute: jest.fn((_data: any, cb: any) => {
					setImmediate(() => cb(null));
				}),
				closeSync: jest.fn(),
			};

			mockClient.setAutoCommit = jest.fn();
			mockClient.commit = jest.fn((callback: any) => {
				if (callback) setImmediate(() => callback(null));
			});
			mockClient.rollback = jest.fn((callback: any) => {
				if (callback) setImmediate(() => callback(null));
			});
			mockClient.prepare = jest.fn((_sql: string, cb: any) => {
				setImmediate(() => cb(null, mockStatement));
			});
		});

		it('should insert vectors with documents', async () => {
			const vectors = [
				[0.1, 0.2, 0.3],
				[0.4, 0.5, 0.6],
			];
			const documents = [
				{ pageContent: 'doc1', metadata: { source: 'test1' } },
				{ pageContent: 'doc2', metadata: { source: 'test2' } },
			];

			await vectorStore.addVectors(vectors, documents);

			// Should execute INSERT for each vector using query (row-by-row)
			expect(mockClient.query).toHaveBeenCalled();
			expect(mockClient.prepare).not.toHaveBeenCalled();
		});

		it('should handle metadata correctly', async () => {
			const vectors = [[0.1, 0.2, 0.3]];
			const documents = [
				{
					pageContent: 'test',
					metadata: {
						source: 'file.txt',
						author: 'John Doe',
						tags: ['tag1', 'tag2'],
					},
				},
			];

			await vectorStore.addVectors(vectors, documents);

			// Should include metadata in query
			expect(mockClient.query).toHaveBeenCalled();
		});

		it('should handle documents without metadata', async () => {
			const vectors = [[0.1, 0.2, 0.3]];
			const documents = [{ pageContent: 'test', metadata: {} }];

			await vectorStore.addVectors(vectors, documents);

			expect(mockClient.query).toHaveBeenCalled();
		});

		it('should throw error if vectors and documents length mismatch', async () => {
			const vectors = [[0.1, 0.2, 0.3]];
			const documents = [
				{ pageContent: 'doc1', metadata: {} },
				{ pageContent: 'doc2', metadata: {} },
			];

			await expect(vectorStore.addVectors(vectors, documents)).rejects.toThrow(
				'Number of vectors and documents must match',
			);
		});
	});

	describe('addDocuments', () => {
		beforeEach(async () => {
			vectorStore = new DB2VectorStore(mockEmbeddings, {
				client: mockClient,
				embeddingFunction: mockEmbeddings,
				tableName: 'test_vectors',
				useBatchInsert: false, // Use row-by-row for these tests
			});

			mockEmbeddings.embedQuery.mockResolvedValue([0.1, 0.2, 0.3]);
			await vectorStore.initialize();
			jest.clearAllMocks();

			// Re-create essential mocks after clearing
			const mockStatement = {
				execute: jest.fn((_data: any, cb: any) => {
					setImmediate(() => cb(null));
				}),
				closeSync: jest.fn(),
			};

			mockClient.setAutoCommit = jest.fn();
			mockClient.commit = jest.fn((callback: any) => {
				if (callback) setImmediate(() => callback(null));
			});
			mockClient.rollback = jest.fn((callback: any) => {
				if (callback) setImmediate(() => callback(null));
			});
			mockClient.prepare = jest.fn((_sql: string, cb: any) => {
				setImmediate(() => cb(null, mockStatement));
			});
		});

		it('should embed documents and add vectors', async () => {
			const documents: Document[] = [
				{ pageContent: 'doc1', metadata: { source: 'test1' } },
				{ pageContent: 'doc2', metadata: { source: 'test2' } },
			];

			const embeddings = [
				[0.1, 0.2, 0.3],
				[0.4, 0.5, 0.6],
			];

			mockEmbeddings.embedDocuments.mockResolvedValue(embeddings);

			await vectorStore.addDocuments(documents);

			// Should call embedDocuments
			expect(mockEmbeddings.embedDocuments).toHaveBeenCalledWith(['doc1', 'doc2']);

			// Should insert vectors
			expect(mockClient.query).toHaveBeenCalled();
		});
	});

	describe('addTexts', () => {
		beforeEach(async () => {
			vectorStore = new DB2VectorStore(mockEmbeddings, {
				client: mockClient,
				embeddingFunction: mockEmbeddings,
				tableName: 'test_vectors',
				useBatchInsert: false, // Use row-by-row for these tests
			});

			mockEmbeddings.embedQuery.mockResolvedValue([0.1, 0.2, 0.3]);
			await vectorStore.initialize();
			jest.clearAllMocks();

			// Re-create essential mocks after clearing
			const mockStatement = {
				execute: jest.fn((_data: any, cb: any) => {
					setImmediate(() => cb(null));
				}),
				closeSync: jest.fn(),
			};

			mockClient.setAutoCommit = jest.fn();
			mockClient.commit = jest.fn((callback: any) => {
				if (callback) setImmediate(() => callback(null));
			});
			mockClient.rollback = jest.fn((callback: any) => {
				if (callback) setImmediate(() => callback(null));
			});
			mockClient.prepare = jest.fn((_sql: string, cb: any) => {
				setImmediate(() => cb(null, mockStatement));
			});
		});

		it('should convert texts to documents and add them', async () => {
			const texts = ['text1', 'text2'];
			const metadatas = [{ source: 'test1' }, { source: 'test2' }];

			const embeddings = [
				[0.1, 0.2, 0.3],
				[0.4, 0.5, 0.6],
			];

			mockEmbeddings.embedDocuments.mockResolvedValue(embeddings);

			await vectorStore.addTexts(texts, metadatas);

			// Should embed texts
			expect(mockEmbeddings.embedDocuments).toHaveBeenCalledWith(texts);

			// Should insert vectors
			expect(mockClient.query).toHaveBeenCalled();
		});

		it('should handle texts without metadata', async () => {
			const texts = ['text1', 'text2'];

			const embeddings = [
				[0.1, 0.2, 0.3],
				[0.4, 0.5, 0.6],
			];

			mockEmbeddings.embedDocuments.mockResolvedValue(embeddings);

			await vectorStore.addTexts(texts);

			expect(mockClient.query).toHaveBeenCalled();
		});
	});

	describe('similaritySearchVectorWithScore', () => {
		beforeEach(async () => {
			vectorStore = new DB2VectorStore(mockEmbeddings, {
				client: mockClient,
				embeddingFunction: mockEmbeddings,
				tableName: 'test_vectors',
				distanceStrategy: DistanceStrategy.EUCLIDEAN,
			});

			mockEmbeddings.embedQuery.mockResolvedValue([0.1, 0.2, 0.3]);
			await vectorStore.initialize();
			jest.clearAllMocks();
		});

		it('should search for similar vectors', async () => {
			const queryVector = [0.1, 0.2, 0.3];
			const k = 5;

			const mockResults = [
				{
					TEXT: 'doc1',
					METADATA: JSON.stringify({ source: 'test1' }),
					DISTANCE: 0.5,
				},
				{
					TEXT: 'doc2',
					METADATA: JSON.stringify({ source: 'test2' }),
					DISTANCE: 0.8,
				},
			];

			mockClient.query.mockImplementation((_sql: string, paramsOrCallback: any, callback?: any) => {
				const cb = typeof paramsOrCallback === 'function' ? paramsOrCallback : callback;
				if (cb) {
					setImmediate(() => cb(null, mockResults));
				}
			});

			const results = await vectorStore.similaritySearchVectorWithScore(queryVector, k);

			// Should execute search query
			expect(mockClient.query).toHaveBeenCalled();

			// Should return documents with scores
			expect(results).toHaveLength(2);
			expect(results[0][0].pageContent).toBe('doc1');
			expect(results[0][1]).toBe(0.5);
		});

		it('should handle empty results', async () => {
			const queryVector = [0.1, 0.2, 0.3];

			mockClient.query.mockImplementation((_sql: string, paramsOrCallback: any, callback?: any) => {
				const cb = typeof paramsOrCallback === 'function' ? paramsOrCallback : callback;
				if (cb) {
					setImmediate(() => cb(null, []));
				}
			});

			const results = await vectorStore.similaritySearchVectorWithScore(queryVector, 5);

			expect(results).toEqual([]);
		});

		it('should parse metadata correctly', async () => {
			const queryVector = [0.1, 0.2, 0.3];

			const mockResults = [
				{
					TEXT: 'test',
					METADATA: JSON.stringify({ source: 'file.txt', author: 'John' }),
					DISTANCE: 0.5,
				},
			];

			mockClient.query.mockImplementation((_sql: string, paramsOrCallback: any, callback?: any) => {
				const cb = typeof paramsOrCallback === 'function' ? paramsOrCallback : callback;
				if (cb) {
					setImmediate(() => cb(null, mockResults));
				}
			});

			const results = await vectorStore.similaritySearchVectorWithScore(queryVector, 1);

			expect(results[0][0].metadata).toEqual({
				source: 'file.txt',
				author: 'John',
			});
		});
	});

	describe('delete', () => {
		beforeEach(async () => {
			vectorStore = new DB2VectorStore(mockEmbeddings, {
				client: mockClient,
				embeddingFunction: mockEmbeddings,
				tableName: 'test_vectors',
			});

			mockEmbeddings.embedQuery.mockResolvedValue([0.1, 0.2, 0.3]);
			await vectorStore.initialize();
			jest.clearAllMocks();
		});

		it('should delete documents by IDs', async () => {
			const ids = ['id1', 'id2', 'id3'];

			await vectorStore.delete({ ids });

			// Should execute DELETE for each ID
			const deleteCalls = (mockClient.query as jest.Mock).mock.calls.filter((call) =>
				call[0].includes('DELETE FROM'),
			);
			expect(deleteCalls.length).toBeGreaterThan(0);
		});

		it('should handle empty ID array', async () => {
			// Empty ID array should throw error
			await expect(vectorStore.delete({ ids: [] })).rejects.toThrow('No IDs provided');
		});
	});

	describe('fromTexts', () => {
		it('should create instance and add texts', async () => {
			const texts = ['text1', 'text2'];
			const metadatas = [{ source: 'test1' }, { source: 'test2' }];
			const embeddings = [
				[0.1, 0.2, 0.3],
				[0.4, 0.5, 0.6],
			];

			mockEmbeddings.embedQuery.mockResolvedValue([0.1, 0.2, 0.3]);
			mockEmbeddings.embedDocuments.mockResolvedValue(embeddings);

			// Set up mocks needed for initialization and insertion
			const mockStatement = {
				execute: jest.fn((_data: any, cb: any) => {
					setImmediate(() => cb(null));
				}),
				closeSync: jest.fn(),
			};

			mockClient.setAutoCommit = jest.fn();
			mockClient.commit = jest.fn((callback: any) => {
				if (callback) setImmediate(() => callback(null));
			});
			mockClient.prepare = jest.fn((_sql: string, cb: any) => {
				setImmediate(() => cb(null, mockStatement));
			});

			const store = await DB2VectorStore.fromTexts(texts, metadatas, mockEmbeddings, {
				client: mockClient,
				tableName: 'test_vectors',
			});

			expect(store).toBeInstanceOf(DB2VectorStore);
			expect(mockEmbeddings.embedDocuments).toHaveBeenCalled();
		});
	});

	describe('fromDocuments', () => {
		it('should create instance and add documents', async () => {
			const documents: Document[] = [
				{ pageContent: 'doc1', metadata: { source: 'test1' } },
				{ pageContent: 'doc2', metadata: { source: 'test2' } },
			];
			const embeddings = [
				[0.1, 0.2, 0.3],
				[0.4, 0.5, 0.6],
			];

			mockEmbeddings.embedQuery.mockResolvedValue([0.1, 0.2, 0.3]);
			mockEmbeddings.embedDocuments.mockResolvedValue(embeddings);

			// Set up mocks needed for initialization and insertion
			const mockStatement = {
				execute: jest.fn((_data: any, cb: any) => {
					setImmediate(() => cb(null));
				}),
				closeSync: jest.fn(),
			};

			mockClient.setAutoCommit = jest.fn();
			mockClient.commit = jest.fn((callback: any) => {
				if (callback) setImmediate(() => callback(null));
			});
			mockClient.prepare = jest.fn((_sql: string, cb: any) => {
				setImmediate(() => cb(null, mockStatement));
			});

			const store = await DB2VectorStore.fromDocuments(documents, mockEmbeddings, {
				client: mockClient,
				tableName: 'test_vectors',
			});

			expect(store).toBeInstanceOf(DB2VectorStore);
			expect(mockEmbeddings.embedDocuments).toHaveBeenCalled();
		});
	});

	describe('distance strategies', () => {
		it('should support euclidean distance', async () => {
			vectorStore = new DB2VectorStore(mockEmbeddings, {
				client: mockClient,
				embeddingFunction: mockEmbeddings,
				tableName: 'test_vectors',
				distanceStrategy: DistanceStrategy.EUCLIDEAN,
			});

			expect(vectorStore).toBeInstanceOf(DB2VectorStore);
		});

		it('should support cosine distance', async () => {
			vectorStore = new DB2VectorStore(mockEmbeddings, {
				client: mockClient,
				embeddingFunction: mockEmbeddings,
				tableName: 'test_vectors',
				distanceStrategy: DistanceStrategy.COSINE,
			});

			expect(vectorStore).toBeInstanceOf(DB2VectorStore);
		});

		it('should support dot product', async () => {
			vectorStore = new DB2VectorStore(mockEmbeddings, {
				client: mockClient,
				embeddingFunction: mockEmbeddings,
				tableName: 'test_vectors',
				distanceStrategy: DistanceStrategy.DOT_PRODUCT,
			});

			expect(vectorStore).toBeInstanceOf(DB2VectorStore);
		});

		describe('batch insertion', () => {
			beforeEach(async () => {
				mockEmbeddings.embedQuery.mockResolvedValue([0.1, 0.2, 0.3]);
			});

			it('should use batch insert when enabled', async () => {
				vectorStore = new DB2VectorStore(mockEmbeddings, {
					client: mockClient,
					embeddingFunction: mockEmbeddings,
					tableName: 'test_vectors',
					useBatchInsert: true,
				});

				await vectorStore.initialize();
				jest.clearAllMocks();

				// Mock prepare and execute for batch insertion
				const mockStatement = {
					execute: jest.fn((_data: any, cb: any) => {
						setImmediate(() => cb(null));
					}),
					closeSync: jest.fn(),
				};

				mockClient.prepare = jest.fn((_sql: string, cb: any) => {
					setImmediate(() => cb(null, mockStatement));
				});

				mockClient.commit = jest.fn((cb: any) => {
					setImmediate(() => cb(null));
				});

				mockClient.setAutoCommit = jest.fn();

				const vectors = [
					[0.1, 0.2, 0.3],
					[0.4, 0.5, 0.6],
				];
				const documents = [
					{ pageContent: 'doc1', metadata: { source: 'test1' } },
					{ pageContent: 'doc2', metadata: { source: 'test2' } },
				];

				await vectorStore.addVectors(vectors, documents);

				// Should use prepare/execute pattern for batch
				expect(mockClient.prepare).toHaveBeenCalled();
				expect(mockStatement.execute).toHaveBeenCalled();
				expect(mockStatement.closeSync).toHaveBeenCalled();
				expect(mockClient.commit).toHaveBeenCalled();
			});

			it('should use row-by-row insert when disabled', async () => {
				vectorStore = new DB2VectorStore(mockEmbeddings, {
					client: mockClient,
					embeddingFunction: mockEmbeddings,
					tableName: 'test_vectors',
					useBatchInsert: false,
				});

				await vectorStore.initialize();

				// Clear all mocks including prepare spy
				jest.clearAllMocks();

				// Re-create the prepare mock as a fresh spy after clearing
				mockClient.prepare = jest.fn();

				mockClient.setAutoCommit = jest.fn();
				mockClient.commit = jest.fn((cb: any) => {
					setImmediate(() => cb(null));
				});

				const vectors = [
					[0.1, 0.2, 0.3],
					[0.4, 0.5, 0.6],
				];
				const documents = [
					{ pageContent: 'doc1', metadata: { source: 'test1' } },
					{ pageContent: 'doc2', metadata: { source: 'test2' } },
				];

				await vectorStore.addVectors(vectors, documents);

				// Should use query, not prepare
				expect(mockClient.query).toHaveBeenCalled();
				expect(mockClient.prepare).not.toHaveBeenCalled();
				expect(mockClient.commit).toHaveBeenCalled();
			});

			it('should default to batch insert when not specified', async () => {
				vectorStore = new DB2VectorStore(mockEmbeddings, {
					client: mockClient,
					embeddingFunction: mockEmbeddings,
					tableName: 'test_vectors',
					// useBatchInsert not specified - should default to true
				});

				await vectorStore.initialize();
				jest.clearAllMocks();

				const mockStatement = {
					execute: jest.fn((_data: any, cb: any) => {
						setImmediate(() => cb(null));
					}),
					closeSync: jest.fn(),
				};

				mockClient.prepare = jest.fn((_sql: string, cb: any) => {
					setImmediate(() => cb(null, mockStatement));
				});

				mockClient.commit = jest.fn((cb: any) => {
					setImmediate(() => cb(null));
				});

				mockClient.setAutoCommit = jest.fn();

				const vectors = [
					[0.1, 0.2, 0.3],
					[0.4, 0.5, 0.6],
				];
				const documents = [
					{ pageContent: 'doc1', metadata: {} },
					{ pageContent: 'doc2', metadata: {} },
				];

				await vectorStore.addVectors(vectors, documents);

				// Should default to batch insert
				expect(mockClient.prepare).toHaveBeenCalled();
			});

			it('should handle batch insert errors with rollback', async () => {
				vectorStore = new DB2VectorStore(mockEmbeddings, {
					client: mockClient,
					embeddingFunction: mockEmbeddings,
					tableName: 'test_vectors',
					useBatchInsert: true,
				});

				mockEmbeddings.embedQuery.mockResolvedValue([0.1, 0.2, 0.3]);
				await vectorStore.initialize();

				// set up error-throwing mock AFTER initialization
				const mockErrorStatement = {
					execute: jest.fn((_data: any, cb: any) => {
						setImmediate(() => cb(new Error('Batch insert failed')));
					}),
					closeSync: jest.fn(),
				};

				// Replace mocks with error versions
				mockClient.setAutoCommit = jest.fn();
				mockClient.prepare = jest.fn((_sql: string, cb: any) => {
					setImmediate(() => cb(null, mockErrorStatement));
				});
				mockClient.commit = jest.fn((cb: any) => {
					setImmediate(() => cb(null));
				});
				mockClient.rollback = jest.fn((cb: any) => {
					setImmediate(() => cb(null));
				});

				const vectors = [
					[0.1, 0.2, 0.3],
					[0.4, 0.5, 0.6],
				];
				const documents = [
					{ pageContent: 'doc1', metadata: {} },
					{ pageContent: 'doc2', metadata: {} },
				];

				await expect(vectorStore.addVectors(vectors, documents)).rejects.toThrow();

				// Should call rollback on error
				expect(mockClient.rollback).toHaveBeenCalled();
				expect(mockErrorStatement.closeSync).toHaveBeenCalled();
			});
		});
	});
});

// Made with Bob
