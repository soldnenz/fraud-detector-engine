"""
FastAPI Service for Fraud Detection
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fraud.inference import FraudModel

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection service",
    version="1.0.0"
)

# Global model instance
model: Optional[FraudModel] = None


class TransactionRequest(BaseModel):
    """Transaction scoring request"""
    # Basic fields
    cst_id: str = Field(..., description="Customer ID")
    target_id: str = Field(..., description="Target/recipient ID")
    amount: float = Field(..., description="Transaction amount")
    
    # Optional fields with defaults
    last_phone_model: Optional[str] = Field("Unknown", description="Last phone model")
    last_os: Optional[str] = Field("Unknown", description="Last OS")
    os_ver_count_30d: Optional[float] = Field(1, description="OS version count 30d")
    phone_model_count_30d: Optional[float] = Field(1, description="Phone model count 30d")
    amount_cum_count: Optional[int] = Field(0, description="Amount cum count")
    
    # Offline features (should be fetched from feature store in production)
    cst_amount_mean_past: Optional[float] = Field(100000, description="Customer's historical mean amount")
    cst_txn_count_past: Optional[int] = Field(0, description="Customer's past transaction count")
    amount_over_mean_past: Optional[float] = Field(1.0, description="Amount ratio vs customer mean")
    amount_diff_mean_past: Optional[float] = Field(0, description="Amount diff vs mean")
    target_fraud_rate_past_smooth: Optional[float] = Field(0.01, description="Target's smoothed fraud rate")
    target_fraud_rate_past: Optional[float] = Field(0.01, description="Target's fraud rate (unsmoothed)")
    target_txn_count_past: Optional[int] = Field(0, description="Target's past transaction count")
    target_txn_count_past_log1p: Optional[float] = Field(0, description="Log of target's past count")
    is_new_target_for_client: Optional[int] = Field(0, description="Is new target for customer")
    is_new_phone_model_for_client: Optional[int] = Field(0, description="Is new phone for customer")
    is_new_os_for_client: Optional[int] = Field(0, description="Is new OS for customer")
    is_first_tx_for_client: Optional[int] = Field(0, description="Is first transaction for customer")
    is_night_tx: Optional[int] = Field(0, description="Is night transaction")
    is_weekend: Optional[int] = Field(0, description="Is weekend transaction")
    hours_since_prev_trans: Optional[float] = Field(999999, description="Hours since previous transaction")
    target_same_as_prev: Optional[int] = Field(0, description="Same target as previous")
    cst_new_targets_ratio: Optional[float] = Field(0, description="New targets ratio")
    cst_night_tx_share: Optional[float] = Field(0, description="Night tx share")
    cst_weekend_tx_share: Optional[float] = Field(0, description="Weekend tx share")
    
    # Behavior features
    sessions_unique_7d: Optional[float] = Field(0, description="Unique sessions in 7 days")
    sessions_unique_30d: Optional[float] = Field(0, description="Unique sessions in 30 days")
    daily_logins_avg_7d: Optional[float] = Field(0, description="Average daily logins in 7 days")
    daily_logins_avg_30d: Optional[float] = Field(0, description="Average daily logins in 30 days")
    
    # Additional behavior features (can be null)
    os_ver_count_30d: Optional[float] = Field(1, description="OS version count in 30 days")
    phone_model_count_30d: Optional[float] = Field(1, description="Phone model count in 30 days")
    login_freq_change_7_vs_30: Optional[float] = Field(0, description="Login frequency change")
    login_share_7_of_30: Optional[float] = Field(0, description="Login share 7d of 30d")
    avg_interval_30d: Optional[float] = Field(0, description="Average interval in 30 days")
    std_interval_30d: Optional[float] = Field(0, description="Std interval in 30 days")
    var_interval_30d: Optional[float] = Field(0, description="Variance interval in 30 days")
    ewm_interval_7d: Optional[float] = Field(0, description="EWM interval in 7 days")
    burstiness: Optional[float] = Field(0, description="Burstiness metric")
    fano_factor: Optional[float] = Field(0, description="Fano factor")
    zscore_interval_7_vs_30: Optional[float] = Field(0, description="Z-score interval 7 vs 30")
    
    class Config:
        schema_extra = {
            "example": {
                "cst_id": "CST12345",
                "target_id": "TGT67890",
                "amount": 500000,
                "last_phone_model": "iPhone 13",
                "last_os": "iOS 15",
                "cst_amount_mean_past": 250000,
                "cst_txn_count_past": 10,
                "amount_over_mean_past": 2.0,
                "target_fraud_rate_past_smooth": 0.005,
                "target_txn_count_past": 50,
                "is_new_target_for_client": 1,
                "is_new_phone_model_for_client": 0,
                "is_night_tx": 0,
                "is_weekend": 0,
                "sessions_unique_7d": 5,
                "sessions_unique_30d": 15,
                "daily_logins_avg_7d": 2.5,
                "daily_logins_avg_30d": 2.0
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_name: str
    model_version: str
    cv_auc: float


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model
    
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data/models/ultimate_v8/model_meta.json"
    )
    
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found at {model_path}")
        print("   Run training first: python train.py")
        return
    
    try:
        model = FraudModel(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_name": model.meta["model_name"],
        "model_version": model.meta["version"],
        "cv_auc": model.meta["cv_auc"]
    }


@app.post("/score")
async def score_transaction(request: TransactionRequest):
    """
    Score a transaction for fraud risk.
    
    Returns decision (BLOCK/REVIEW/ACCEPT) with probability and explanation.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to dict
        tx_row = request.model_dump()
        
        # Score transaction
        result = model.score_transaction(tx_row, explain=True)
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.post("/score_batch")
async def score_batch(transactions: list[TransactionRequest]):
    """
    Score multiple transactions in batch.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for tx in transactions:
            tx_row = tx.model_dump()
            result = model.score_transaction(tx_row, explain=False)
            result["cst_id"] = tx.cst_id
            result["target_id"] = tx.target_id
            result["amount"] = tx.amount
            results.append(result)
        
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

