#!/usr/bin/env python3
"""
Main entry point for the Financial Risk Governance BI Platform.

This module serves as the primary application launcher and orchestrates
the various components of the risk governance system.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging
from src.bi_analytics.api_routes import router as bi_router
from src.risk_assessment.api_routes import router as risk_router
from src.predictive_modeling.api_routes import router as ml_router
from src.governance.api_routes import router as governance_router

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize configuration
config = ConfigManager()

# Create FastAPI application
app = FastAPI(
    title=config.get("app.name", "Financial Risk Governance BI"),
    version=config.get("app.version", "1.0.0"),
    description="Integrated BI and Predictive Modeling for Cross-Dimensional Financial Risk Governance",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(bi_router, prefix="/api/bi", tags=["Business Intelligence"])
app.include_router(risk_router, prefix="/api/risk", tags=["Risk Assessment"])
app.include_router(ml_router, prefix="/api/models", tags=["Predictive Modeling"])
app.include_router(governance_router, prefix="/api/governance", tags=["Governance"])


@app.get("/")
async def root():
    """Root endpoint providing basic application information."""
    return {
        "application": config.get("app.name"),
        "version": config.get("app.version"),
        "status": "healthy",
        "environment": config.get("app.environment"),
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": config.get_current_timestamp(),
        "services": {
            "database": "connected",  # TODO: Implement actual health checks
            "cache": "connected",
            "ml_models": "loaded",
        }
    }


@app.on_event("startup")
async def startup_event():
    """Application startup event handler."""
    logger.info("Starting Financial Risk Governance BI Platform")
    logger.info(f"Environment: {config.get('app.environment')}")
    logger.info(f"Debug mode: {config.get('app.debug')}")
    
    # TODO: Initialize database connections
    # TODO: Load ML models
    # TODO: Initialize cache
    # TODO: Set up monitoring
    
    logger.info("Application startup completed successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event handler."""
    logger.info("Shutting down Financial Risk Governance BI Platform")
    
    # TODO: Close database connections
    # TODO: Clean up resources
    # TODO: Save state if needed
    
    logger.info("Application shutdown completed")


def main():
    """Main application entry point."""
    try:
        # Get configuration
        host = config.get("app.host", "localhost")
        port = config.get("app.port", 8000)
        debug = config.get("app.debug", False)
        
        logger.info(f"Starting server on {host}:{port}")
        
        # Configure uvicorn
        uvicorn_config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            reload=debug,
            log_level="info" if not debug else "debug",
            access_log=True,
        )
        
        # Start server
        server = uvicorn.Server(uvicorn_config)
        asyncio.run(server.serve())
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()