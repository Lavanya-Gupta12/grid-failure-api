from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from supabase import create_client, Client
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Power Grid Failure Prediction API",
    description="Predict grid failures using GNN+LSTM+XGBoost ensemble",
    version="1.0.0"
)

# CORS - Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your Lovable URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# ============================================================================
# PYDANTIC MODELS (Request/Response schemas)
# ============================================================================

class NodeRisk(BaseModel):
    node_id: int
    risk_score: float
    risk_tier: str
    latitude: float
    longitude: float
    name: str
    type: Optional[str] = None
    age_years: Optional[int] = None

class NodeDetail(BaseModel):
    node_id: int
    name: str
    latitude: float
    longitude: float
    type: str
    voltage_kv: int
    capacity_mw: float
    age_years: int
    last_maintenance_months: int
    maintenance_quality: str
    risk_score: float
    risk_tier: str

class MaintenanceTask(BaseModel):
    id: int
    node_id: int
    priority: int
    scheduled_date: str
    status: str
    risk_score: float

class FeatureImportance(BaseModel):
    feature_name: str
    importance_value: float

class Stats(BaseModel):
    total_nodes: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    pending_maintenance: int
    avg_risk_score: float

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def assign_risk_tier(risk_score: float) -> str:
    """Assign risk tier based on score"""
    if risk_score >= 0.7:
        return "High"
    elif risk_score >= 0.4:
        return "Medium"
    else:
        return "Low"

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Power Grid Failure Prediction API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/api/health")
async def health_check():
    """Check if Supabase connection works"""
    try:
        response = supabase.table("grid_nodes").select("count", count="exact").limit(1).execute()
        return {
            "status": "healthy",
            "database": "connected",
            "nodes_count": response.count
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/api/nodes", response_model=List[NodeRisk])
async def get_all_nodes(
    risk_tier: Optional[str] = None,
    limit: int = 1000,
    offset: int = 0
):
    """
    Get all nodes with risk scores
    
    - **risk_tier**: Filter by 'High', 'Medium', or 'Low' (optional)
    - **limit**: Number of results (default 1000)
    - **offset**: Pagination offset (default 0)
    """
    try:
        # Query predictions
        pred_query = supabase.table("predictions")\
            .select("*")\
            .order("risk_score", desc=True)\
            .limit(limit)\
            .offset(offset)
        
        if risk_tier:
            pred_query = pred_query.eq("risk_tier", risk_tier)
        
        pred_response = pred_query.execute()
        
        # Get unique node_ids
        node_ids = [item["node_id"] for item in pred_response.data]
        
        # Query grid_nodes for those IDs
        nodes_response = supabase.table("grid_nodes")\
            .select("node_id, name, latitude, longitude, type, age_years")\
            .in_("node_id", node_ids)\
            .execute()
        
        # Create a lookup dictionary
        nodes_dict = {n["node_id"]: n for n in nodes_response.data}
        
        # Combine the data
        nodes = []
        for pred in pred_response.data:
            node_data = nodes_dict.get(pred["node_id"])
            if node_data:
                nodes.append(NodeRisk(
                    node_id=pred["node_id"],
                    risk_score=pred["risk_score"],
                    risk_tier=pred["risk_tier"],
                    latitude=node_data["latitude"],
                    longitude=node_data["longitude"],
                    name=node_data["name"],
                    type=node_data.get("type"),
                    age_years=node_data.get("age_years")
                ))
        
        return nodes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching nodes: {str(e)}")
    
@app.get("/api/node/{node_id}")
async def get_node_details(node_id: int):
    """Get detailed information for a specific node"""
    try:
        # Get node + prediction
        node_response = supabase.table("grid_nodes")\
            .select("*, predictions(*)")\
            .eq("node_id", node_id)\
            .execute()
        
        if not node_response.data:
            raise HTTPException(status_code=404, detail="Node not found")
        
        node_data = node_response.data[0]
        prediction_data = node_data.get("predictions", [{}])[0] if node_data.get("predictions") else {}
        
        # Get top 10 important features
        features_response = supabase.table("feature_importance")\
            .select("feature_name, importance_value")\
            .eq("node_id", node_id)\
            .order("importance_value", desc=True)\
            .limit(10)\
            .execute()
        
        return {
            "node": {
                "node_id": node_data["node_id"],
                "name": node_data["name"],
                "latitude": node_data["latitude"],
                "longitude": node_data["longitude"],
                "type": node_data.get("type"),
                "voltage_kv": node_data.get("voltage_kv"),
                "capacity_mw": node_data.get("capacity_mw"),
                "age_years": node_data.get("age_years"),
                "last_maintenance_months": node_data.get("last_maintenance_months"),
                "maintenance_quality": node_data.get("maintenance_quality"),
                "degree": node_data.get("degree")
            },
            "prediction": {
                "risk_score": prediction_data.get("risk_score"),
                "risk_tier": prediction_data.get("risk_tier"),
                "graphsage_score": prediction_data.get("graphsage_score"),
                "lstm_score": prediction_data.get("lstm_score"),
                "xgboost_score": prediction_data.get("xgboost_score"),
                "prediction_timestamp": prediction_data.get("prediction_timestamp")
            },
            "top_features": [
                FeatureImportance(
                    feature_name=f["feature_name"],
                    importance_value=f["importance_value"]
                ) for f in features_response.data
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching node details: {str(e)}")

@app.get("/api/maintenance", response_model=List[MaintenanceTask])
async def get_maintenance_schedule(
    status: Optional[str] = "pending",
    limit: int = 100
):
    """Get maintenance schedule"""
    try:
        query = supabase.table("maintenance_schedule")\
            .select("*")\
            .order("priority", desc=False)\
            .limit(limit)
        
        if status:
            query = query.eq("status", status)
        
        response = query.execute()
        
        return [
            MaintenanceTask(
                id=item["id"],
                node_id=item["node_id"],
                priority=item["priority"],
                scheduled_date=item["scheduled_date"],
                status=item["status"],
                risk_score=item["risk_score"]
            )
            for item in response.data
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching maintenance: {str(e)}")

@app.post("/api/maintenance/{task_id}/complete")
async def complete_maintenance(task_id: int):
    """Mark a maintenance task as completed"""
    try:
        response = supabase.table("maintenance_schedule")\
            .update({
                "status": "completed",
                "updated_at": datetime.now().isoformat()
            })\
            .eq("id", task_id)\
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {
            "message": "Maintenance task marked as completed",
            "task_id": task_id,
            "data": response.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating task: {str(e)}")

@app.get("/api/stats", response_model=Stats)
async def get_statistics():
    """Get overall grid statistics"""
    try:
        # Count by risk tier
        high = supabase.table("predictions").select("node_id", count="exact").eq("risk_tier", "High").execute()
        medium = supabase.table("predictions").select("node_id", count="exact").eq("risk_tier", "Medium").execute()
        low = supabase.table("predictions").select("node_id", count="exact").eq("risk_tier", "Low").execute()
        
        # Pending maintenance
        pending = supabase.table("maintenance_schedule").select("id", count="exact").eq("status", "pending").execute()
        
        # Average risk score
        all_predictions = supabase.table("predictions").select("risk_score").execute()
        avg_risk = np.mean([p["risk_score"] for p in all_predictions.data]) if all_predictions.data else 0
        
        return Stats(
            total_nodes=high.count + medium.count + low.count,
            high_risk_count=high.count,
            medium_risk_count=medium.count,
            low_risk_count=low.count,
            pending_maintenance=pending.count,
            avg_risk_score=float(avg_risk)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")

# ============================================================================
# ADMIN ENDPOINT - DATA POPULATION
# ============================================================================

@app.post("/api/admin/populate")
async def populate_database():
    """
    ONE-TIME SETUP: Populate database with data from Person A
    """
    try:
        print("Clearing existing data...")
        supabase.table("feature_importance").delete().neq("id", 0).execute()
        supabase.table("maintenance_schedule").delete().neq("id", 0).execute()
        supabase.table("predictions").delete().neq("id", 0).execute()
        supabase.table("grid_nodes").delete().neq("node_id", 0).execute()
        print("✅ Existing data cleared")
    
        
        # Load CSVs
        print("Loading CSV files...")
        predictions_df = pd.read_csv("predictions_for_person_b.csv")
        features_df = pd.read_csv("features_v4.csv")
        
        print(f"Predictions columns: {predictions_df.columns.tolist()}")
        print(f"Features columns: {features_df.columns.tolist()}")
        
        # Merge data
        data = predictions_df.merge(features_df, on="node_id", how="inner")
        print(f"Merged data: {len(data)} rows")
        print(f"Merged columns: {data.columns.tolist()}")
        
        # ========================================================================
        # INSERT NODES - WITH FLEXIBLE COLUMN MAPPING
        # ========================================================================
        print("Inserting grid nodes...")
        nodes_data = []
        
        for _, row in data.iterrows():
            # Handle column name variations
            lat = row.get("latitude") or row.get("lat") or row.get("Latitude") or 19.0760
            lon = row.get("longitude") or row.get("lon") or row.get("Longitude") or 72.8777
            
            nodes_data.append({
                "node_id": int(row["node_id"]),
                "name": str(row.get("name", f"Node_{int(row['node_id']):04d}")),
                "latitude": float(lat),
                "longitude": float(lon),
                "type": str(row.get("type", "distribution")),
                "usage": str(row.get("usage", "residential")),
                "voltage_kv": int(row.get("voltage_kv", 11)),
                "capacity_mw": float(row.get("capacity_mw", 50.0)),
                "age_years": int(row.get("age_years", 10)),
                "urban": str(row.get("urban", "urban")),
                "last_maintenance_months": int(row.get("last_maintenance_months", 6)),
                "maintenance_quality": str(row.get("maintenance_quality", "average")),
                "degree": int(row.get("degree", 5))
            })
        
        # Insert in batches
        for i in range(0, len(nodes_data), 1000):
            batch = nodes_data[i:i+1000]
            supabase.table("grid_nodes").insert(batch).execute()
            print(f"  Inserted nodes batch {i//1000 + 1}")
        
        # ========================================================================
        # INSERT PREDICTIONS
        # ========================================================================
        print("Inserting predictions...")
        predictions_data = []
        
        for _, row in data.iterrows():
            # Try different column name variations
            ensemble_score = (
                row.get("ensemble_weighted_prediction") or 
                row.get("ensemble_prediction") or 
                row.get("risk_score") or 
                0.5
            )
            
            risk_score = float(ensemble_score)
            
            predictions_data.append({
                "node_id": int(row["node_id"]),
                "risk_score": risk_score,
                "risk_tier": assign_risk_tier(risk_score),
                "graphsage_score": float(row.get("graphsage_prediction", 0.0)),
                "lstm_score": float(row.get("lstm_prediction", 0.0)),
                "xgboost_score": float(row.get("xgboost_prediction", 0.0)),
                "model_version": "ensemble_v1"
            })
        
        for i in range(0, len(predictions_data), 1000):
            batch = predictions_data[i:i+1000]
            supabase.table("predictions").insert(batch).execute()
            print(f"  Inserted predictions batch {i//1000 + 1}")
        
        # ========================================================================
        # GENERATE MAINTENANCE SCHEDULE
        # ========================================================================
        print("Generating maintenance schedule...")
        
        # Sort by risk score (whatever column it is)
        risk_col = "ensemble_weighted_prediction"
        if risk_col not in data.columns:
            risk_col = "ensemble_prediction" if "ensemble_prediction" in data.columns else "risk_score"
        
        high_risk = data[data[risk_col] >= 0.7].sort_values(risk_col, ascending=False)
        
        maintenance_data = []
        start_date = datetime.now() + timedelta(days=1)
        
        for idx, (_, row) in enumerate(high_risk.head(50).iterrows()):
            maintenance_data.append({
                "node_id": int(row["node_id"]),
                "priority": min((idx // 10) + 1, 5),
                "scheduled_date": (start_date + timedelta(days=idx)).date().isoformat(),
                "risk_score": float(row[risk_col]),
                "status": "pending",
                "notes": f"High risk node (score: {row[risk_col]:.3f})"
            })
        
        if maintenance_data:
            supabase.table("maintenance_schedule").insert(maintenance_data).execute()
        
        # ========================================================================
        # INSERT FEATURE IMPORTANCE (Dummy data)
        # ========================================================================
        print("Inserting feature importance...")
        feature_names = [
            "age_years", "load_mean", "voltage_dev_max", "last_maintenance_months",
            "capacity_mw", "degree", "utilization_rate", "stress_duration_pct"
        ]
        
        features_data = []
        for node_id in high_risk.head(100)["node_id"]:
            for i, feat in enumerate(feature_names):
                features_data.append({
                    "node_id": int(node_id),
                    "feature_name": feat,
                    "importance_value": float(np.random.random() * (10 - i)),
                    "explanation_type": "SHAP"
                })
        
        if features_data:
            for i in range(0, len(features_data), 1000):
                batch = features_data[i:i+1000]
                supabase.table("feature_importance").insert(batch).execute()
        
        print("✅ Database population complete!")
        
        return {
            "status": "success",
            "message": "Database populated successfully",
            "nodes_inserted": len(nodes_data),
            "predictions_inserted": len(predictions_data),
            "maintenance_tasks_created": len(maintenance_data)
        }
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail="CSV files not found. Make sure predictions_for_person_b.csv and features_v4.csv are in the same folder."
        )
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required column in CSV: {str(e)}. Please check CSV structure."
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)