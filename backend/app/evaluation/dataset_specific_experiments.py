"""Isolated FD002--FD004 experimental evaluation helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
from app.evaluation.real_only_evaluation import calculate_binary_metrics, fit_future_degradation_target
from app.models.real_only_anomaly import RealOnlyAnomalyPipeline
from app.preparation.early_features import EarlyWindowConfig, extract_early_features
from app.preparation.splitting import split_by_engine

def feature_table(frame):
    result=extract_early_features(frame,EarlyWindowConfig(1,30,30))
    if result.features.shape[1]!=169: raise ValueError("Expected existing 168 features.")
    return result.features,result.skipped_engines
def official_test_truth(test_frame,rul_values,target):
    ids=sorted(int(value) for value in test_frame["engine_id"].unique())
    if len(ids)!=len(rul_values): raise ValueError("Official RUL value count does not match TEST engines.")
    truth=test_frame.groupby("engine_id",as_index=False)["cycle"].max().sort_values("engine_id");truth["rul"]=np.asarray(rul_values,dtype=float)
    if not np.isfinite(truth["rul"]).all() or (truth["rul"]<0).any(): raise ValueError("RUL values must be finite and non-negative.")
    truth["remaining_cycles"]=truth["cycle"]-target.early_end_cycle+truth["rul"];truth["near_term_degradation"]=truth["remaining_cycles"]<=target.remaining_cycle_horizon
    return truth.set_index("engine_id")
def train_dataset_specific(train_frame):
    split=split_by_engine(train_frame,seed=42);sets=[set(x.engine_id) for x in (split.train,split.validation,split.test)]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)): raise AssertionError("Engine-level split overlaps.")
    train,skip_train=feature_table(split.train);validation,skip_validation=feature_table(split.validation)
    if skip_train or skip_validation: raise ValueError("Training/validation engines need 30 cycles.")
    target=fit_future_degradation_target(split.train,early_end_cycle=30,training_quantile=.20);pipeline=RealOnlyAnomalyPipeline().fit(train,validation);ends=split.validation.groupby("engine_id",as_index=False)["cycle"].max().set_index("engine_id");y=ends.loc[validation.engine_id,"cycle"].to_numpy(float)-30<=target.remaining_cycle_horizon;scores=pipeline.predict(validation).risk_score.to_numpy(float)
    return pipeline,target,split.metadata,calculate_binary_metrics(y,scores>=pipeline.thresholds.watchlist,scores)
def evaluate_official_test(pipeline,test_frame,rul_values,target):
    features,skipped=feature_table(test_frame);p=pipeline.predict(features).set_index("engine_id");truth=official_test_truth(test_frame,rul_values,target).loc[p.index];score=p.risk_score.to_numpy(float);counts={n:int((p.classification==n).sum()) for n in ("Normal","Watchlist","High Risk")}
    dist=lambda x:{k:float(v) for k,v in x.agg(["min","median","mean","max"]).items()}
    return {"metrics":calculate_binary_metrics(truth.near_term_degradation.to_numpy(bool),score>=pipeline.thresholds.watchlist,score),"evaluated_engine_count":len(p),"positive_count":int(truth.near_term_degradation.sum()),"negative_count":int((~truth.near_term_degradation).sum()),"excluded":skipped,"status_counts":counts,"risk_score_distribution":dist(p.risk_score),"z_score_distribution":dist(p.z_score),"isolation_score_distribution":dist(p.isolation_score)}
