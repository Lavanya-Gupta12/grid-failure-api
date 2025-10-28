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
from sklearn.preprocessing import LabelEncoder
from lime.lime_tabular import LimeTabularExplainer
import shap

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
# GLOBAL EXPLAINER VARIABLES
# ============================================================================

explainer_lime = None
explainer_shap = None
explainer_X = None
explainer_feature_names = None
explainer_merged_df = None

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

def initialize_explainers():
    """Initialize LIME and SHAP explainers on startup"""
    global explainer_lime, explainer_shap, explainer_X, explainer_feature_names, explainer_merged_df
    
    try:
        print("🔄 Initializing LIME and SHAP explainers...")
        
        # Load data
        features_df = pd.read_csv("features_v4.csv")
        predictions_df = pd.read_csv("predictions_for_person_b.csv")
        
        print(f"📊 Predictions columns: {predictions_df.columns.tolist()}")
        
        # Select prediction columns (use actual CSV column names)
        pred_cols = ['node_id']
        
        # Map CSV columns to standard names
        rename_map = {}
        
        if 'graphsage_pred' in predictions_df.columns:
            pred_cols.append('graphsage_pred')
            rename_map['graphsage_pred'] = 'graphsage_prediction'
        
        if 'lstm_pred' in predictions_df.columns:
            pred_cols.append('lstm_pred')
            rename_map['lstm_pred'] = 'lstm_prediction'
        
        if 'xgboost_pred' in predictions_df.columns:
            pred_cols.append('xgboost_pred')
            rename_map['xgboost_pred'] = 'xgboost_prediction'
        
        if 'ensemble_pred' in predictions_df.columns:
            pred_cols.append('ensemble_pred')
            rename_map['ensemble_pred'] = 'ensemble_weighted_prediction'
        
        # Select and rename columns
        pred_subset = predictions_df[pred_cols].copy()
        pred_subset = pred_subset.rename(columns=rename_map)
        
        # Merge datasets
        explainer_merged_df = features_df.merge(
            pred_subset,
            on='node_id',
            how='inner'
        )
        
        print(f"✓ Merged {len(explainer_merged_df)} rows")
        
        # Prepare features
        exclude_cols = [
            'node_id', 'name', 'latitude', 'longitude', 'type', 'usage', 'urban',
            'graphsage_prediction', 'lstm_prediction', 'xgboost_prediction', 
            'ensemble_weighted_prediction', 'risk_score', 'ensemble_prediction',
            'lat', 'lon', 'Latitude', 'Longitude'  # Additional variations
        ]
        
        feature_cols = [c for c in explainer_merged_df.columns if c not in exclude_cols]
        X_df = explainer_merged_df[feature_cols].copy()
        
        # Encode categorical columns
        for col in X_df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))
        
        explainer_X = X_df.values
        explainer_feature_names = X_df.columns.tolist()
        
        # Initialize LIME
        explainer_lime = LimeTabularExplainer(
            explainer_X,
            feature_names=explainer_feature_names,
            class_names=['No Failure', 'Failure'],
            discretize_continuous=True,
            mode='classification'
        )
        print("✓ LIME explainer initialized")
        
        # Initialize SHAP
        background_data = shap.sample(explainer_X, 20)
        
        def ensemble_predict(X_batch):
            w_graphsage = 0.333
            w_lstm = 0.331
            w_xgboost = 0.336
            
            predictions = []
            for row in X_batch:
                distances = np.sum(np.abs(explainer_X - row), axis=1)
                idx = np.argmin(distances)
                
                ensemble_prob = (
                    w_graphsage * explainer_merged_df.iloc[idx]['graphsage_prediction'] +
                    w_lstm * explainer_merged_df.iloc[idx]['lstm_prediction'] +
                    w_xgboost * explainer_merged_df.iloc[idx]['xgboost_prediction']
                )
                predictions.append(ensemble_prob)
            
            return np.array(predictions)
        
        explainer_shap = shap.KernelExplainer(ensemble_predict, background_data)
        print("✓ SHAP explainer initialized")
        print("✅ All explainers ready!")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize explainers: {str(e)}")
        print("Explainability endpoints will not be available")

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
            "nodes_count": response.count,
            "explainers_initialized": explainer_lime is not None and explainer_shap is not None
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

@app.get("/api/explain/{node_id}")
async def explain_node(node_id: int):
    """
    Get LIME and SHAP explanations for a specific node.
    
    This endpoint provides explainability for the risk prediction:
    - **LIME**: Local interpretable model-agnostic explanations
    - **SHAP**: SHapley Additive exPlanations
    
    Returns the top 10 features influencing the failure prediction for the specified node.
    """
    try:
        if explainer_lime is None or explainer_shap is None:
            raise HTTPException(status_code=503, detail="Explainers not initialized")
        
        # Find node in data
        node_data = explainer_merged_df[explainer_merged_df['node_id'] == node_id]
        if node_data.empty:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
        
        node_row = node_data.iloc[0]
        row_idx = node_data.index[0]
        array_idx = explainer_merged_df.index.get_loc(row_idx)
        
        # Define prediction function for LIME
        def ensemble_predict_proba(X_batch):
            w_graphsage = 0.333
            w_lstm = 0.331
            w_xgboost = 0.336
            predictions = []
            for row in X_batch:
                distances = np.sum(np.abs(explainer_X - row), axis=1)
                idx = np.argmin(distances)
                ensemble_prob = (
                    w_graphsage * explainer_merged_df.iloc[idx]['graphsage_prediction'] +
                    w_lstm * explainer_merged_df.iloc[idx]['lstm_prediction'] +
                    w_xgboost * explainer_merged_df.iloc[idx]['xgboost_prediction']
                )
                predictions.append([1 - ensemble_prob, ensemble_prob])
            return np.array(predictions)
        
        # Generate LIME explanation
        lime_exp = explainer_lime.explain_instance(
            explainer_X[array_idx],
            ensemble_predict_proba,
            num_features=10
        )
        lime_features = [
            {'feature': f, 'weight': float(w)} 
            for f, w in lime_exp.as_list()
        ]
        
        # Generate SHAP explanation
        shap_values = explainer_shap.shap_values(explainer_X[array_idx:array_idx+1])[0]
        feature_importance = list(zip(explainer_feature_names, shap_values))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        shap_features = [
            {'feature': f, 'value': float(v)} 
            for f, v in feature_importance[:10]
        ]
        
        # Build response
        response = {
            'node_id': int(node_id),
            'name': str(node_row['name']),
            'ensemble_prediction': float(node_row['ensemble_weighted_prediction']),
            'risk_tier': assign_risk_tier(float(node_row['ensemble_weighted_prediction'])),
            'lime': {
                'features': lime_features
            },
            'shap': {
                'features': shap_features,
                'base_value': float(explainer_shap.expected_value)
            }
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")

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

# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize explainers when app starts"""
    initialize_explainers()

# ============================================================================
# MAIN
# ============================================================================
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)