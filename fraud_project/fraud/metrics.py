"""
Metrics Module
Handles model metrics and business metrics
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    confusion_matrix,
)


@dataclass
class BusinessMetrics:
    total_fraud_amount: float
    blocked_fraud_amount: float
    missed_fraud_amount: float
    blocked_legit_amount: float
    fraud_prevention_rate: float
    total_cost: float
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Metrics:
    roc_auc: float
    precision: float
    recall: float
    f1: float
    fbeta_05: float
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    business_metrics: Optional[BusinessMetrics] = None

    def to_dict(self):
        d = asdict(self)
        if self.business_metrics:
            d['business_metrics'] = self.business_metrics.to_dict()
        return d


def compute_class_weight(y: pd.Series) -> float:
    """Compute class weight for imbalanced data"""
    pos = float(y.sum())
    neg = float(len(y) - pos)
    return neg / (pos + 1e-6)


def pick_best_threshold_by_f1(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """Find threshold that maximizes F1 score"""
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    f1_arr = 2 * prec * rec / (prec + rec + 1e-9)
    if len(thr) == 0:
        return 0.5
    best_idx = int(np.argmax(f1_arr[:-1]))
    return float(thr[best_idx])


def pick_threshold_by_cost(
    y_true: pd.Series,
    y_proba: np.ndarray,
    amounts: pd.Series,
    fp_cost_ratio: float = 0.1,
    fn_cost_ratio: float = 1.0,
) -> float:
    """
    Find threshold that minimizes business cost.
    FP cost = fp_cost_ratio * amount (friction cost)
    FN cost = fn_cost_ratio * amount (actual fraud loss)
    """
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    
    best_threshold = 0.5
    best_cost = float('inf')
    
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        
        # Calculate costs
        fraud_mask = y_true == 1
        pred_fraud_mask = y_pred == 1
        
        # FN: missed frauds
        fn_mask = fraud_mask & (~pred_fraud_mask)
        fn_cost = (amounts[fn_mask] * fn_cost_ratio).sum()
        
        # FP: blocked legit
        fp_mask = (~fraud_mask) & pred_fraud_mask
        fp_cost = (amounts[fp_mask] * fp_cost_ratio).sum()
        
        total_cost = fn_cost + fp_cost
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = float(thr)
    
    return best_threshold


def compute_business_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    amounts: pd.Series,
    fp_cost_ratio: float = 0.1,
    fn_cost_ratio: float = 1.0,
) -> BusinessMetrics:
    """Compute business-oriented metrics"""
    fraud_mask = y_true == 1
    pred_fraud_mask = y_pred == 1
    
    total_fraud_amount = amounts[fraud_mask].sum()
    
    tp_mask = fraud_mask & pred_fraud_mask
    blocked_fraud_amount = amounts[tp_mask].sum()
    
    fn_mask = fraud_mask & (~pred_fraud_mask)
    missed_fraud_amount = amounts[fn_mask].sum()
    
    fp_mask = (~fraud_mask) & pred_fraud_mask
    blocked_legit_amount = amounts[fp_mask].sum()
    
    fraud_prevention_rate = (
        blocked_fraud_amount / total_fraud_amount if total_fraud_amount > 0 else 0.0
    )
    
    # Total business cost
    fn_cost = missed_fraud_amount * fn_cost_ratio
    fp_cost = blocked_legit_amount * fp_cost_ratio
    total_cost = fn_cost + fp_cost
    
    return BusinessMetrics(
        total_fraud_amount=float(total_fraud_amount),
        blocked_fraud_amount=float(blocked_fraud_amount),
        missed_fraud_amount=float(missed_fraud_amount),
        blocked_legit_amount=float(blocked_legit_amount),
        fraud_prevention_rate=float(fraud_prevention_rate),
        total_cost=float(total_cost),
    )


def evaluate_predictions(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    amounts: Optional[pd.Series] = None,
    beta: float = 0.5,
    fp_cost_ratio: float = 0.1,
    fn_cost_ratio: float = 1.0,
) -> Metrics:
    """Evaluate predictions with given threshold"""
    y_pred = (y_proba >= threshold).astype(int)
    roc = roc_auc_score(y_true, y_proba)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fbeta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    business_metrics = None
    if amounts is not None:
        business_metrics = compute_business_metrics(
            y_true, y_pred, amounts, fp_cost_ratio, fn_cost_ratio
        )
    
    return Metrics(
        roc_auc=roc,
        precision=prec,
        recall=rec,
        f1=f1,
        fbeta_05=fbeta,
        threshold=float(threshold),
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        business_metrics=business_metrics,
    )


def print_metrics(label: str, m: Metrics):
    """Print metrics in human-readable format"""
    print(
        f"[{label}] ROC-AUC: {m.roc_auc:.3f} | "
        f"P: {m.precision:.3f} | R: {m.recall:.3f} | "
        f"F1: {m.f1:.3f} | F0.5: {m.fbeta_05:.3f} | thr: {m.threshold:.4f}"
    )
    print(f"    TP={m.tp}, FP={m.fp}, TN={m.tn}, FN={m.fn}")
    if m.business_metrics:
        bm = m.business_metrics
        print(f"    💰 Blocked fraud: ₸{bm.blocked_fraud_amount:,.0f} ({bm.fraud_prevention_rate*100:.1f}%)")
        print(f"    📊 Total cost: ₸{bm.total_cost:,.0f}")

