"""Validation-only FD002--FD004 fusion-weight experiment; no artifacts written."""
from __future__ import annotations
import argparse,csv,hashlib
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from app.evaluation.dataset_specific_experiments import feature_table,official_test_truth,train_dataset_specific
from app.evaluation.real_only_evaluation import calculate_binary_metrics,construct_future_degradation_truth
from app.preparation.cmapss_adapter import CMAPSSSubset,CMAPSSVariant,load_cmapss,load_cmapss_rul
from app.preparation.splitting import split_by_engine
WEIGHTS=tuple(round(i/10,1) for i in range(11))
def load(zip_path,name,variant,subset):
    with ZipFile(zip_path) as z:
        n=next(x for x in z.namelist() if Path(x).name.lower()==name.lower())
        with z.open(n) as h:return load_cmapss(h,variant=variant,subset=subset) if subset else load_cmapss_rul(h)
def fuse(pred,z):return np.clip(z*pred.z_risk.to_numpy(float)+(1-z)*pred.isolation_risk.to_numpy(float),0,1)
def assess(scores,truth,watch,high):
    m=calculate_binary_metrics(truth,scores>=watch,scores); s=np.where(scores>=high,"High Risk",np.where(scores>=watch,"Watchlist","Normal"));c={x:int((s==x).sum()) for x in ("Normal","Watchlist","High Risk")};return m,c
def training_partitions(train_frame, metadata):
    """Rebuild the deterministic TRAIN-only split represented by metadata."""
    split = split_by_engine(train_frame, seed=metadata.seed)
    if split.metadata != metadata: raise AssertionError("Rebuilt TRAIN split does not match training metadata.")
    return split
def main():
    p=argparse.ArgumentParser();p.add_argument("--zip",type=Path,default=Path(r"C:\Users\Deepak Roy\OneDrive\Desktop\CMAPSSData.zip"));p.add_argument("--output",type=Path,default=Path(__file__).with_name("ml_fusion_weight_diagnostic_results.csv"));a=p.parse_args();prod=Path(__file__).resolve().parent/"model_artifacts/burnui_production_model.joblib";before=hashlib.sha256(prod.read_bytes()).hexdigest().upper();rows=[];frozen=[]
    # Selection phase: TRAIN and validation only; official TEST/RUL never loaded here.
    for name in ("FD002","FD003","FD004"):
        v=CMAPSSVariant(name);train=load(a.zip,f"train_{name}.txt",v,CMAPSSSubset.TRAIN);pipe,target,metadata,_=train_dataset_specific(train);split=training_partitions(train,metadata);feat,_=feature_table(split.validation);pred=pipe.predict(feat);truth=construct_future_degradation_truth(split.validation,target).set_index("engine_id").loc[pred.engine_id,"near_term_degradation"].to_numpy(bool);candidates=[]
        for z in WEIGHTS:
            score=fuse(pred,z);watch,high=float(np.quantile(score,.85)),float(np.quantile(score,.95));m,c=assess(score,truth,watch,high);row={"dataset":name,"scope":"VALIDATION","z_weight":z,"isolation_weight":1-z,"watchlist_threshold":watch,"high_risk_threshold":high,"engine_count":len(score),"alert_count":c["Watchlist"]+c["High Risk"],"normal_count":c["Normal"],"watchlist_count":c["Watchlist"],"high_risk_count":c["High Risk"],**m};candidates.append(row);rows.append(row)
        eligible=[x for x in candidates if x["alert_count"]<=x["engine_count"]*.5];pool=eligible or candidates;selected=max(pool,key=lambda x:(x["recall"] or 0,x["f1"] or 0,x["precision"] or 0,-(x["fpr"] or 1)));selected={**selected,"selected":True,"alert_constraint_fallback":not bool(eligible)};frozen.append((name,v,pipe,target,split,selected));
    # Only after selection is locked are official TEST and RUL opened.
    for name,v,pipe,target,split,selected in frozen:
        hold,_=feature_table(split.test);hold_pred=pipe.predict(hold);hold_truth=construct_future_degradation_truth(split.test,target).set_index("engine_id").loc[hold_pred.engine_id,"near_term_degradation"].to_numpy(bool);hold_score=fuse(hold_pred,selected["z_weight"]);m,c=assess(hold_score,hold_truth,selected["watchlist_threshold"],selected["high_risk_threshold"]);rows.append({**selected,"scope":"INTERNAL_HOLDOUT_LOCKED","engine_count":len(hold_score),"alert_count":c["Watchlist"]+c["High Risk"],"normal_count":c["Normal"],"watchlist_count":c["Watchlist"],"high_risk_count":c["High Risk"],**m})
        test=load(a.zip,f"test_{name}.txt",v,CMAPSSSubset.TEST);rul=load(a.zip,f"RUL_{name}.txt",v,None);feat,_=feature_table(test);pred=pipe.predict(feat);truth=official_test_truth(test,rul,target).loc[pred.engine_id,"near_term_degradation"].to_numpy(bool);score=fuse(pred,selected["z_weight"]);m,c=assess(score,truth,selected["watchlist_threshold"],selected["high_risk_threshold"]);rows.append({**selected,"scope":"OFFICIAL_TEST_LOCKED","engine_count":len(score),"alert_count":c["Watchlist"]+c["High Risk"],"normal_count":c["Normal"],"watchlist_count":c["Watchlist"],"high_risk_count":c["High Risk"],**m})
    fields=sorted({k for x in rows for k in x});
    with a.output.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
    after=hashlib.sha256(prod.read_bytes()).hexdigest().upper();assert before==after;print(f"FD001 SHA before/after: {before}");print(f"Wrote {a.output}")
if __name__=="__main__":main()
