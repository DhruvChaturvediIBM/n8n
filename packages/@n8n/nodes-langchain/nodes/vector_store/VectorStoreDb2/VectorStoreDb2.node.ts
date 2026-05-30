import { NodeOperationError, type INodeProperties } from 'n8n-workflow';
import { createVectorStoreNode, metadataFilterField } from '@n8n/ai-utilities';
import { DB2VectorStore } from './utils/db2VectorStore';
import type { DistanceStrategy } from './utils/types';

const sharedFields: INodeProperties[] = [
	{
		displayName: 'Table Name',
		name: 'tableName',
		type: 'string',
		default: 'vector_store',
		required: true,
		description: 'Name of the DB2 table to store vectors',
	},
	{
		displayName: 'Distance Strategy',
		name: 'distanceStrategy',
		type: 'options',
		default: 'euclidean',
		description: 'Strategy for calculating distance between vectors',
		options: [
			{
				name: 'Euclidean',
				value: 'euclidean',
			},
			{
				name: 'Cosine',
				value: 'cosine',
			},
			{
				name: 'Dot Product',
				value: 'dot_product',
			},
		],
	},
];

const insertFields: INodeProperties[] = [
	{
		displayName: 'Options',
		name: 'options',
		type: 'collection',
		placeholder: 'Add Option',
		default: {},
		options: [
			{
				displayName: 'Use Batch Insert',
				name: 'useBatchInsert',
				type: 'boolean',
				default: true,
				description:
					'Whether to use batch insertion for better performance. Disable for row-by-row insertion.',
			},
		],
	},
];

const retrieveFields: INodeProperties[] = [
	{
		displayName: 'Options',
		name: 'options',
		type: 'collection',
		placeholder: 'Add Option',
		default: {},
		options: [metadataFilterField],
	},
];

// Connection pool to reuse DB2 connections
const connectionPool = new Map<string, any>();

/**
 * Build DB2 connection configuration object
 * Using object format instead of connection string to avoid credential exposure in logs
 */
function buildConnectionConfig(credentials: any): any {
	const config: any = {
		DATABASE: credentials.database,
		HOSTNAME: credentials.host,
		PORT: credentials.port,
		PROTOCOL: 'TCPIP',
		UID: credentials.user,
		PWD: credentials.password,
	};

	// Add SSL/TLS configuration if enabled
	if (credentials.ssl) {
		config.Security = 'SSL';
		if (credentials.sslCertificate) {
			// SSL certificate should be provided as a file path by the user
			// The certificate file must be accessible to the n8n process
			config.SSLServerCertificate = credentials.sslCertificate;
		}
	}

	// Add connection timeout
	if (credentials.connectionTimeout) {
		config.ConnectTimeout = credentials.connectionTimeout;
	}

	return config;
}

/**
 * Get or create a DB2 connection from the pool
 */
async function getConnection(credentials: any): Promise<any> {
	const config = buildConnectionConfig(credentials);
	const poolKey = `${credentials.host}:${credentials.port}:${credentials.database}:${credentials.user}`;

	// Check if we have a valid connection in the pool
	let client = connectionPool.get(poolKey);

	if (client) {
		// Test if connection is still alive
		try {
			await new Promise((resolve, reject) => {
				client.query('SELECT 1 FROM SYSIBM.SYSDUMMY1', (err: Error) => {
					if (err) reject(err);
					else resolve(null);
				});
			});
			return client;
		} catch (error) {
			// Connection is dead, remove from pool
			connectionPool.delete(poolKey);
			try {
				client.close(() => {});
			} catch {}
		}
	}

	// Create new connection
	// eslint-disable-next-line @typescript-eslint/no-require-imports
	const ibmDb = require('ibm_db');

	// Convert config object to connection string for ibm_db
	const connStr = Object.entries(config)
		.map(([key, value]) => `${key}=${value}`)
		.join(';');

	client = await new Promise<any>((resolve, reject) => {
		ibmDb.open(connStr, (err: Error, conn: any) => {
			if (err) {
				reject(new Error(`Failed to connect to DB2: ${err.message}`));
			} else {
				resolve(conn);
			}
		});
	});

	// Store in pool
	connectionPool.set(poolKey, client);

	return client;
}

export class VectorStoreDb2 extends createVectorStoreNode({
	meta: {
		displayName: 'DB2 Vector Store',
		name: 'vectorStoreDb2',
		description: 'Work with IBM DB2 Vector Store for embeddings and similarity search',
		icon: 'file:Db2.svg',
		docsUrl:
			'https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.vectorstoredb2/',
		credentials: [
			{
				name: 'db2Api',
				required: true,
			},
		],
		operationModes: ['load', 'insert', 'retrieve', 'retrieve-as-tool'],
	},
	sharedFields,
	insertFields,
	loadFields: retrieveFields,
	retrieveFields,
	updateFields: [],
	async getVectorStoreClient(context, _filter, embeddings, itemIndex) {
		const credentials = await context.getCredentials('db2Api');
		const tableName = context.getNodeParameter('tableName', itemIndex) as string;
		const distanceStrategy = context.getNodeParameter(
			'distanceStrategy',
			itemIndex,
		) as DistanceStrategy;

		try {
			// Get connection from pool
			const client = await getConnection(credentials);

			return await DB2VectorStore.fromExistingIndex(embeddings, {
				client,
				tableName,
				distanceStrategy,
			});
		} catch (error) {
			throw new NodeOperationError(
				context.getNode(),
				`Failed to initialize DB2 Vector Store: ${error instanceof Error ? error.message : 'Unknown error'}`,
				{ itemIndex },
			);
		}
	},
	async populateVectorStore(context, embeddings, documents, itemIndex) {
		const credentials = await context.getCredentials('db2Api');
		const tableName = context.getNodeParameter('tableName', itemIndex) as string;
		const distanceStrategy = context.getNodeParameter(
			'distanceStrategy',
			itemIndex,
		) as DistanceStrategy;
		const useBatchInsert = context.getNodeParameter(
			'options.useBatchInsert',
			itemIndex,
			true,
		) as boolean;

		try {
			// Get connection from pool
			const client = await getConnection(credentials);

			const vectorStore = new DB2VectorStore(embeddings, {
				client,
				tableName,
				distanceStrategy,
				embeddingFunction: embeddings,
				useBatchInsert,
			});

			await vectorStore.initialize();
			await vectorStore.addDocuments(documents);
		} catch (error) {
			throw new NodeOperationError(
				context.getNode(),
				`Failed to populate DB2 Vector Store: ${error instanceof Error ? error.message : 'Unknown error'}`,
				{ itemIndex },
			);
		}
	},
	releaseVectorStoreClient(_vectorStore) {
		// Connections are managed by the pool and reused
		// They will be closed when the process exits or on connection errors
		// Individual vector store instances don't own the connection
	},
}) {}

// Made with Bob
