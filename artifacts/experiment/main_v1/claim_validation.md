# Claim validation

| Claim | Metric | Observed | Verdict |
|---|---|---|---|
| Routine proxies beat random capacity but miss most cases | recall at 10% capacity | random 0.100; routine 0.163 | supported |
| School-experience scales add information | paired school bootstrap, expanded logit minus routine logit | AUPRC +0.104 (95% interval +0.090 to +0.118); recall +0.104 (+0.085 to +0.122) | supported |
| Nonlinear model has a modest positive gain | paired school bootstrap, expanded HGB minus expanded logit; researcher-set descriptive margins +0.02 AUPRC and +0.03 recall | AUPRC +0.026 (+0.015 to +0.036); recall +0.016 (+0.003 to +0.029) | paired gain above zero; lower interval bounds below chosen margins; no cost-based practical claim |
| Average performance transports across Spanish regions | region IECV versus school nested | AUPRC 0.323 versus 0.329; regional recall 0.213–0.394 and calibration slope 0.598–1.362 | supported only at pooled level; local heterogeneity is material |
| Main information-tier result survives sensitivity analyses | continuous outcome, complete cases, and school-local capacity | expanded information improved continuous RMSE; complete-case AUPRC 0.327; local-capacity recall 0.267 | supported |
| Audit attributes should be ranking inputs | gain and recall-gap changes after adding sex, immigration and ESCS | AUPRC +0.020; recall-gap increases: sex +0.182, immigration +0.043, ESCS +0.076 | refuted for the primary operational model; retain for auditing |

The statistical campaign is complete within the frozen scope. Claims remain conditional on rendered table/figure QA and are not prospective, causal, diagnostic or deployment claims.
