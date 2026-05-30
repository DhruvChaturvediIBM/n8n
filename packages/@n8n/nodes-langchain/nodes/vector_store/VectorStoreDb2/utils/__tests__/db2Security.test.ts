import {
	validateDatabaseName,
	validateHostname,
	validatePort,
	validateIdentifier,
	getQuotedIdentifier,
	sanitizeSqlString,
	createSafeErrorMessage,
	validateConnectionConfig,
	validateTableName,
} from '../db2Security';

describe('db2Security', () => {
	describe('validateDatabaseName', () => {
		it('should accept valid database names', () => {
			expect(validateDatabaseName('mydb')).toBe('mydb');
			expect(validateDatabaseName('DB2_TEST')).toBe('DB2_TEST');
			expect(validateDatabaseName('  mydb  ')).toBe('mydb'); // trims whitespace
		});

		it('should reject non-string values', () => {
			expect(() => validateDatabaseName(123)).toThrow('must be a string');
			expect(() => validateDatabaseName(null)).toThrow('must be a string');
			expect(() => validateDatabaseName(undefined)).toThrow('must be a string');
		});

		it('should reject empty strings', () => {
			expect(() => validateDatabaseName('')).toThrow('cannot be empty');
			expect(() => validateDatabaseName('   ')).toThrow('cannot be empty');
		});

		it('should reject names exceeding maximum length', () => {
			const longName = 'a'.repeat(129);
			expect(() => validateDatabaseName(longName)).toThrow('exceeds maximum length');
		});

		it('should reject names with unsafe characters', () => {
			expect(() => validateDatabaseName('my"db')).toThrow('unsafe characters');
			expect(() => validateDatabaseName("my'db")).toThrow('unsafe characters');
			expect(() => validateDatabaseName('my;db')).toThrow('unsafe characters');
			expect(() => validateDatabaseName('my\\db')).toThrow('unsafe characters');
			expect(() => validateDatabaseName('my\ndb')).toThrow('unsafe characters');
			expect(() => validateDatabaseName('my\rdb')).toThrow('unsafe characters');
			expect(() => validateDatabaseName('my\tdb')).toThrow('unsafe characters');
		});
	});

	describe('validateHostname', () => {
		it('should accept valid hostnames', () => {
			expect(validateHostname('localhost')).toBe('localhost');
			expect(validateHostname('db.example.com')).toBe('db.example.com');
			expect(validateHostname('192.168.1.1')).toBe('192.168.1.1');
			expect(validateHostname('my-server')).toBe('my-server');
		});

		it('should reject non-string values', () => {
			expect(() => validateHostname(123)).toThrow('must be a string');
		});

		it('should reject empty strings', () => {
			expect(() => validateHostname('')).toThrow('cannot be empty');
		});

		it('should reject hostnames with unsafe characters', () => {
			expect(() => validateHostname('host"name')).toThrow('unsafe characters');
			expect(() => validateHostname("host'name")).toThrow('unsafe characters');
			expect(() => validateHostname('host;name')).toThrow('unsafe characters');
			expect(() => validateHostname('host\\name')).toThrow('unsafe characters');
			expect(() => validateHostname('host/name')).toThrow('unsafe characters');
			expect(() => validateHostname('host?name')).toThrow('unsafe characters');
			expect(() => validateHostname('host#name')).toThrow('unsafe characters');
			expect(() => validateHostname('host name')).toThrow('unsafe characters');
		});

		it('should reject malformed hostnames', () => {
			expect(() => validateHostname('..hostname')).toThrow('malformed hostname');
			expect(() => validateHostname('.hostname')).toThrow('malformed hostname');
			expect(() => validateHostname('-hostname')).toThrow('malformed hostname');
			expect(() => validateHostname('hostname.')).toThrow('malformed hostname');
			expect(() => validateHostname('host..name')).toThrow('malformed hostname');
		});

		it('should reject hostnames with unsupported characters', () => {
			expect(() => validateHostname('host_name')).toThrow('unsupported characters');
			expect(() => validateHostname('host@name')).toThrow('unsupported characters');
		});
	});

	describe('validatePort', () => {
		it('should accept valid port numbers', () => {
			expect(validatePort(1)).toBe(1);
			expect(validatePort(80)).toBe(80);
			expect(validatePort(443)).toBe(443);
			expect(validatePort(50000)).toBe(50000);
			expect(validatePort(65535)).toBe(65535);
		});

		it('should reject non-number values', () => {
			expect(() => validatePort('80')).toThrow('must be a number');
			expect(() => validatePort(null)).toThrow('must be a number');
			expect(() => validatePort(undefined)).toThrow('must be a number');
			expect(() => validatePort(true)).toThrow('must be a number');
		});

		it('should reject non-integer values', () => {
			expect(() => validatePort(80.5)).toThrow('must be an integer');
			expect(() => validatePort(NaN)).toThrow('must be an integer');
		});

		it('should reject out-of-range ports', () => {
			expect(() => validatePort(0)).toThrow('must be between 1 and 65535');
			expect(() => validatePort(-1)).toThrow('must be between 1 and 65535');
			expect(() => validatePort(65536)).toThrow('must be between 1 and 65535');
			expect(() => validatePort(100000)).toThrow('must be between 1 and 65535');
		});
	});

	describe('validateIdentifier', () => {
		it('should accept valid identifiers', () => {
			expect(validateIdentifier('table1')).toBe('table1');
			expect(validateIdentifier('my_table')).toBe('my_table');
			expect(validateIdentifier('Table123')).toBe('Table123');
			expect(validateIdentifier('T')).toBe('T');
		});

		it('should reject non-string values', () => {
			expect(() => validateIdentifier(123)).toThrow('must be a string');
		});

		it('should reject empty strings', () => {
			expect(() => validateIdentifier('')).toThrow('cannot be empty');
		});

		it('should reject identifiers starting with numbers', () => {
			expect(() => validateIdentifier('1table')).toThrow('starting with a letter');
		});

		it('should reject identifiers starting with underscores', () => {
			expect(() => validateIdentifier('_table')).toThrow('starting with a letter');
		});

		it('should reject identifiers with special characters', () => {
			expect(() => validateIdentifier('table-name')).toThrow('letters, numbers, and underscores');
			expect(() => validateIdentifier('table.name')).toThrow('letters, numbers, and underscores');
			expect(() => validateIdentifier('table name')).toThrow('letters, numbers, and underscores');
			expect(() => validateIdentifier('table@name')).toThrow('letters, numbers, and underscores');
		});

		it('should reject identifiers exceeding maximum length', () => {
			const longName = 'a' + 'b'.repeat(128);
			expect(() => validateIdentifier(longName)).toThrow('letters, numbers, and underscores');
		});

		it('should use custom field name in error messages', () => {
			expect(() => validateIdentifier('123', 'column name')).toThrow('Invalid column name');
		});
	});

	describe('getQuotedIdentifier', () => {
		it('should quote simple identifiers', () => {
			expect(getQuotedIdentifier('table1')).toBe('"table1"');
			expect(getQuotedIdentifier('my_table')).toBe('"my_table"');
		});

		it('should escape existing double quotes', () => {
			expect(getQuotedIdentifier('my"table')).toBe('"my""table"');
			expect(getQuotedIdentifier('"table"')).toBe('"""table"""');
		});

		it('should handle multiple double quotes', () => {
			expect(getQuotedIdentifier('my"test"table')).toBe('"my""test""table"');
		});
	});

	describe('sanitizeSqlString', () => {
		it('should escape single quotes', () => {
			expect(sanitizeSqlString("O'Brien")).toBe("O''Brien");
			expect(sanitizeSqlString("It's")).toBe("It''s");
		});

		it('should handle multiple single quotes', () => {
			expect(sanitizeSqlString("'test'")).toBe("''test''");
			expect(sanitizeSqlString("a'b'c")).toBe("a''b''c");
		});

		it('should not modify strings without single quotes', () => {
			expect(sanitizeSqlString('normal text')).toBe('normal text');
			expect(sanitizeSqlString('test123')).toBe('test123');
		});

		it('should handle empty strings', () => {
			expect(sanitizeSqlString('')).toBe('');
		});
	});

	describe('createSafeErrorMessage', () => {
		it('should create basic error messages', () => {
			const error = new Error('Connection failed');
			const message = createSafeErrorMessage(error);
			expect(message).toBe('DB2 operation failed: Connection failed');
		});

		it('should include context when provided', () => {
			const error = new Error('Query failed');
			const message = createSafeErrorMessage(error, 'during insert');
			expect(message).toBe('DB2 operation failed during insert: Query failed');
		});

		it('should redact password information', () => {
			const error = new Error('Connection failed: PWD=secret123');
			const message = createSafeErrorMessage(error);
			expect(message).toContain('[REDACTED]');
			expect(message).not.toContain('secret123');
		});

		it('should redact various sensitive patterns', () => {
			const patterns = [
				'PWD=secret',
				'PASSWORD=secret',
				'UID=user123',
				'USER=user123',
				'USERID=user123',
				'HOSTNAME=db.example.com',
				'DATABASE=mydb',
				'PORT=50000',
			];

			for (const pattern of patterns) {
				const error = new Error(`Failed: ${pattern}`);
				const message = createSafeErrorMessage(error);
				expect(message).toContain('[REDACTED]');
				expect(message).not.toContain(pattern.split('=')[1]);
			}
		});

		it('should handle errors without messages', () => {
			const error = new Error();
			const message = createSafeErrorMessage(error);
			expect(message).toBe('DB2 operation failed: Unknown error');
		});
	});

	describe('validateConnectionConfig', () => {
		const validConfig = {
			hostname: 'localhost',
			port: 50000,
			database: 'testdb',
			username: 'user',
			password: 'pass',
		};

		it('should accept valid configuration', () => {
			const result = validateConnectionConfig(validConfig);
			expect(result.valid).toBe(true);
			expect(result.error).toBeUndefined();
		});

		it('should reject invalid hostname', () => {
			const result = validateConnectionConfig({
				...validConfig,
				hostname: 'host name',
			});
			expect(result.valid).toBe(false);
			expect(result.error).toContain('hostname');
		});

		it('should reject invalid port', () => {
			const result = validateConnectionConfig({
				...validConfig,
				port: 0,
			});
			expect(result.valid).toBe(false);
			expect(result.error).toContain('port');
		});

		it('should reject invalid database name', () => {
			const result = validateConnectionConfig({
				...validConfig,
				database: 'db;name',
			});
			expect(result.valid).toBe(false);
			expect(result.error).toContain('database');
		});

		it('should reject empty username', () => {
			const result = validateConnectionConfig({
				...validConfig,
				username: '',
			});
			expect(result.valid).toBe(false);
			expect(result.error).toContain('username');
		});

		it('should reject empty password', () => {
			const result = validateConnectionConfig({
				...validConfig,
				password: '',
			});
			expect(result.valid).toBe(false);
			expect(result.error).toContain('password');
		});
	});

	describe('validateTableName', () => {
		it('should accept valid table names', () => {
			const result = validateTableName('my_table');
			expect(result.valid).toBe(true);
			expect(result.error).toBeUndefined();
		});

		it('should reject invalid table names', () => {
			const result = validateTableName('123table');
			expect(result.valid).toBe(false);
			expect(result.error).toContain('table name');
		});

		it('should reject non-string values', () => {
			const result = validateTableName(123);
			expect(result.valid).toBe(false);
			expect(result.error).toBeDefined();
		});
	});
});

// Made with Bob
