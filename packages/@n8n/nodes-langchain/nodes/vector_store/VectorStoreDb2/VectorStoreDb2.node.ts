import type { INodeProperties } from 'n8n-workflow';
import { NodeOperationError } from 'n8n-workflow';
import { createVectorStoreNode } from '@n8n/ai-utilities';
import { DB2VectorStore } from './utils/db2VectorStore';
import type { DistanceStrategy } from './utils/types';
import * as fs from 'fs';
import * as path from 'path';

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

export class VectorStoreDb2 extends createVectorStoreNode({
	meta: {
		displayName: 'DB2 Vector Store',
		name: 'vectorStoreDb2',
		description: 'Work with IBM DB2 Vector Store for embeddings and similarity search',
		icon: 'file:db2.svg',
		docsUrl:
			'https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.vectorstoredb2/',
		credentials: [
			{
				name: 'db2',
				required: true,
			},
		],
	},
	sharedFields,
	insertFields: [],
	loadFields: [],
	retrieveFields: [],
	updateFields: [],
	async getVectorStoreClient(context, _filter, embeddings, itemIndex) {
		const credentials = await context.getCredentials('db2');
		const tableName = context.getNodeParameter('tableName', itemIndex) as string;
		const distanceStrategy = context.getNodeParameter(
			'distanceStrategy',
			itemIndex,
		) as DistanceStrategy;

		// Create DB2 connection
		const ibmDb = require('ibm_db');

		// Build connection string
		let connStr = `DATABASE=${credentials.database};HOSTNAME=${credentials.host};PORT=${credentials.port};PROTOCOL=TCPIP;UID=${credentials.user};PWD=${credentials.password};`;

		// Add SSL configuration if enabled
		if (credentials.ssl === true) {
			// Support both old and new field names for backward compatibility
			const sslCertPath = (credentials.sslCertificatePath || credentials.sslCertificate) as string;

			// Validate SSL certificate path
			if (!sslCertPath || sslCertPath.trim() === '') {
				throw new NodeOperationError(
					context.getNode(),
					'SSL Certificate Path is required when SSL is enabled',
					{
						itemIndex,
						description: 'Please enter the path to your SSL certificate in the credentials',
					},
				);
			}

			// Check if certificate file exists
			const resolvedPath = path.resolve(sslCertPath);
			if (!fs.existsSync(resolvedPath)) {
				throw new NodeOperationError(
					context.getNode(),
					`SSL Certificate file not found at path: ${resolvedPath}`,
					{
						itemIndex,
						description: 'Please verify the certificate path is correct and the file exists',
					},
				);
			}

			// Check if file is readable
			try {
				fs.accessSync(resolvedPath, fs.constants.R_OK);
			} catch (error) {
				throw new NodeOperationError(
					context.getNode(),
					`SSL Certificate file is not readable: ${resolvedPath}`,
					{
						itemIndex,
						description: 'Please check file permissions',
					},
				);
			}

			// Add SSL parameters to connection string
			connStr += `SECURITY=SSL;SSLServerCertificate=${resolvedPath};`;
		}

		// Attempt connection with error handling
		const client = await new Promise<any>((resolve, reject) => {
			ibmDb.open(connStr, (err: Error, conn: any) => {
				if (err) {
					// Check for SSL-specific errors
					const errorMessage = err.message || err.toString();

					if (errorMessage.includes('SSL') || errorMessage.includes('certificate')) {
						reject(
							new NodeOperationError(context.getNode(), `SSL Connection failed: ${errorMessage}`, {
								itemIndex,
								description:
									'Please verify:\n1. SSL certificate is valid\n2. Certificate path is correct\n3. Certificate matches the server',
							}),
						);
					} else if (errorMessage.includes('authentication') || errorMessage.includes('password')) {
						reject(
							new NodeOperationError(
								context.getNode(),
								'Authentication failed: Invalid username or password',
								{ itemIndex },
							),
						);
					} else if (errorMessage.includes('connection') || errorMessage.includes('timeout')) {
						reject(
							new NodeOperationError(context.getNode(), `Connection failed: ${errorMessage}`, {
								itemIndex,
								description:
									'Please verify:\n1. Host and port are correct\n2. Database is accessible\n3. Network connectivity',
							}),
						);
					} else {
						reject(
							new NodeOperationError(context.getNode(), `DB2 connection failed: ${errorMessage}`, {
								itemIndex,
							}),
						);
					}
				} else {
					resolve(conn);
				}
			});
		});

		return await DB2VectorStore.fromExistingIndex(embeddings, {
			client,
			tableName,
			distanceStrategy,
		});
	},
	async populateVectorStore(context, embeddings, documents, itemIndex) {
		const vectorStore = await this.getVectorStoreClient(context, undefined, embeddings, itemIndex);
		await vectorStore.addDocuments(documents);
	},
}) {}

// Made with Bob
