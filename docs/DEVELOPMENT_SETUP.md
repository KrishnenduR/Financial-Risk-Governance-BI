# Development Setup Guide

## Prerequisites

- Python 3.9 or higher
- PostgreSQL 12 or higher
- Redis 6 or higher
- Git (for version control)

## Environment Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd financial-risk-governance-bi
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create a `.env` file in the project root:

```env
# Database Configuration
DB_USERNAME=risk_user
DB_PASSWORD=your_secure_password
DW_USERNAME=warehouse_user
DW_PASSWORD=your_warehouse_password

# Redis Configuration
REDIS_PASSWORD=your_redis_password

# Security
JWT_SECRET_KEY=your_jwt_secret_key_here

# External APIs
BLOOMBERG_API_KEY=your_bloomberg_api_key
```

### 5. Database Setup

#### PostgreSQL Installation and Configuration

1. Install PostgreSQL from https://www.postgresql.org/download/
2. Create databases:

```sql
-- Connect as postgres user
CREATE DATABASE risk_governance;
CREATE DATABASE risk_warehouse;

-- Create users
CREATE USER risk_user WITH PASSWORD 'your_secure_password';
CREATE USER warehouse_user WITH PASSWORD 'your_warehouse_password';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE risk_governance TO risk_user;
GRANT ALL PRIVILEGES ON DATABASE risk_warehouse TO warehouse_user;
```

#### Initialize Database Schema

```bash
python scripts/init_db.py
```

### 6. Redis Setup

#### Install Redis

- **Windows**: Download from https://github.com/microsoftarchive/redis/releases
- **macOS**: `brew install redis`
- **Linux**: `sudo apt-get install redis-server`

#### Start Redis

```bash
redis-server
```

## Running the Application

### Development Mode

```bash
python src/main.py
```

The application will be available at:
- Main application: http://localhost:8000
- API documentation: http://localhost:8000/api/docs
- Health check: http://localhost:8000/health

### Production Mode

```bash
# Set environment to production
export APP_ENVIRONMENT=production

# Run with Gunicorn (install first: pip install gunicorn)
gunicorn src.main:app --workers 4 --bind 0.0.0.0:8000
```

## Testing

### Run Unit Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

### Load Testing

```bash
# Install locust: pip install locust
locust -f tests/load_tests.py --host=http://localhost:8000
```

## Code Quality

### Formatting

```bash
black src/
```

### Linting

```bash
flake8 src/
```

### Type Checking

```bash
mypy src/
```

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add your feature description"

# Push branch
git push origin feature/your-feature-name
```

### 2. Code Review Process

1. Create pull request
2. Ensure all tests pass
3. Code review by team members
4. Address feedback
5. Merge to main branch

### 3. Database Migrations

When modifying database schemas:

```bash
# Create migration
python scripts/create_migration.py "migration_description"

# Apply migrations
python scripts/migrate.py
```

## Docker Development (Optional)

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "src/main.py"]
```

## Debugging

### VS Code Configuration

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/src/main.py",
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    }
  ]
}
```

### Logging

Logs are written to:
- Console (stdout)
- File: `logs/app.log`

Adjust logging level in `config/config.yaml`:

```yaml
logging:
  root:
    level: DEBUG  # Change to DEBUG for verbose logging
```

## Common Issues and Solutions

### Issue: Import Errors

**Solution**: Ensure PYTHONPATH is set correctly:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Database Connection Failed

**Solution**: 
1. Verify PostgreSQL is running
2. Check connection parameters in config
3. Ensure databases exist

### Issue: Redis Connection Failed

**Solution**:
1. Verify Redis is running: `redis-cli ping`
2. Check Redis configuration
3. Verify password if set

### Issue: Module Not Found

**Solution**: Ensure virtual environment is activated and dependencies are installed:
```bash
pip install -r requirements.txt
```

## Performance Optimization

### Database Optimization

1. Add appropriate indexes
2. Use connection pooling
3. Implement query optimization

### Caching Strategy

1. Redis for session data
2. Application-level caching for API responses
3. Database query result caching

### Monitoring

1. Enable Prometheus metrics
2. Set up health checks
3. Monitor application logs

## Contributing

1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass
5. Submit pull request for review

For more information, see the main [README.md](../README.md) file.