## [2.0.0] - 2024-12-30

### Added
- Support for Python 3.13.1 features and optimizations
- Asynchronous SQLite backend using SQLAlchemy 2.0
- Connection pooling with AsyncAdaptedQueuePool
- Automatic connection health checks with pool_pre_ping
- Retry mechanism with exponential backoff for transient errors
- Database indices for commonly queried fields
- Parallel query execution for read operations
- Batch processing for entity and relation operations
- Comprehensive error handling and logging
- Resource cleanup mechanisms

### Changed
- Migrated from JSONL file storage to SQLite database
- Updated database URL format to use aiosqlite
- Improved entity and relation model definitions
- Enhanced transaction management
- Optimized query patterns for better performance
- Modernized type hints and async patterns

### Removed
- JSONL file-based storage system
- Legacy synchronous operations
- Manual index management
- In-memory caching system

### Fixed
- Concurrent access issues with file-based storage
- Memory leaks in long-running operations
- Transaction isolation problems
- Connection handling in error cases

### Security
- Implemented proper SQL query parameterization
- Added transaction isolation improvements
- Enhanced error handling for sensitive operations

### Performance
- Reduced database query count through batch operations
- Improved memory usage with efficient connection pooling
- Optimized search operations with database indices
- Enhanced concurrent operation handling
- Implemented parallel query execution where applicable

### Dependencies
- Added aiosqlite for async SQLite support
- Updated SQLAlchemy to version 2.0
- Added greenlet for SQLAlchemy async support

### Configuration
Added new configuration options:
- DATABASE_URL - Now uses SQLAlchemy async format
- POOL_SIZE - Controls connection pool size
- MAX_OVERFLOW - Sets maximum additional connections
- POOL_TIMEOUT - Connection acquisition timeout
- POOL_RECYCLE - Connection recycling interval

## Migration Guide

### Migrating from 1.x to 2.0.0

1. Database Configuration:
   Update your configuration from:
   ```json
   "DATABASE_URL": "/path/to/file"
   ```
   To:
   ```json
   "DATABASE_URL": "sqlite+aiosqlite:////path/to/database.db"
   ```

2. Additional Configuration:
   Add the following optional parameters:
   ```json
   {
       "POOL_SIZE": "5",
       "MAX_OVERFLOW": "10",
       "POOL_TIMEOUT": "30",
       "POOL_RECYCLE": "3600"
   }
   ```

3. Data Migration:
   Run the provided migration script to transfer existing JSONL data to SQLite:
   ```bash
   python -m memory_mcp_server.migrate_data --source old_data.jsonl --target sqlite:///new_database.db
   ```

