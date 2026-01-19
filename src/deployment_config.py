"""
A/B Testing Deployment Configuration and Instructions
======================================================

This file provides configuration and deployment instructions for Phase 1B
A/B testing system.

Author: AETHER Development Team
Version: 3.0.0 - Phase 1B Deployment
Date: November 25, 2025
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

AB_TESTING_CONFIG = {
    # Enable/disable A/B testing
    "enabled": True,
    
    # Enhanced model allocation (0.0 to 1.0)
    # 0.20 = 20% Enhanced, 80% Legacy
    "enhanced_allocation": 0.20,
    
    # Auto-rollback settings
    "enable_auto_rollback": True,
    "rollback_threshold": 0.02,  # 2% worse Sharpe triggers rollback
    "min_predictions_before_rollback": 100,
    
    # Model paths
    "legacy_model_path": "models/legacy_transformer_phase1b.pth",
    "enhanced_model_path": "models/enhanced_transformer_phase1b.pth",
    
    # Performance tracking
    "save_report_every_n_predictions": 50,
    "report_path": "models/ab_test_report.json",
    
    # Monitoring
    "log_every_n_predictions": 10,
    "print_report_every_n_predictions": 100
}


# =============================================================================
# DEPLOYMENT INSTRUCTIONS
# =============================================================================

DEPLOYMENT_INSTRUCTIONS = """
PHASE 1B: A/B TESTING DEPLOYMENT
=================================

STEP 1: VERIFY MODELS EXIST
----------------------------
Check that both models are in place:
  ✓ models/legacy_transformer_phase1b.pth
  ✓ models/enhanced_transformer_phase1b.pth

If missing, run: python src/tools/train_phase1b.py


STEP 2: CONFIGURE SETTINGS.YAML
--------------------------------
Add to config/settings.yaml:

```yaml
ai_parameters:
  # Phase 1B A/B Testing
  enable_ab_testing: true
  enhanced_allocation: 0.20  # 20% Enhanced, 80% Legacy
  ab_auto_rollback: true
  ab_rollback_threshold: 0.02  # 2% Sharpe difference
```


STEP 3: UPDATE MAIN BOT (src/main_bot.py)
------------------------------------------
Replace NexusBrain import:

OLD:
```python
from src.ai_core.nexus_brain import NexusBrain
```

NEW:
```python
from src.ai_core.nexus_brain_ab import NexusBrainAB as NexusBrain
```

Update initialization:
```python
# In __init__ method
self.nexus_brain = NexusBrain(
    config=self.config,
    enable_ab_testing=self.config.get('ai_parameters', {}).get('enable_ab_testing', True),
    enhanced_allocation=self.config.get('ai_parameters', {}).get('enhanced_allocation', 0.20)
)
```

Add outcome recording after trade execution:
```python
# After trade closes
actual_signal = "BUY" if profit > 0 else "SELL" if profit < 0 else "NEUTRAL"
self.nexus_brain.record_outcome(actual_signal, pnl=profit)
```


STEP 4: MONITOR PERFORMANCE
----------------------------
View real-time performance:
  python src/tools/monitor_ab_test.py

Manual report generation:
  >>> from src.ai_core.nexus_brain_ab import NexusBrainAB
  >>> brain = NexusBrainAB()
  >>> brain.print_performance_report()


STEP 5: GRADUAL ROLLOUT PLAN
-----------------------------
Week 1-2: 20% Enhanced (monitor closely)
Week 3-4: If Enhanced Sharpe ≥2% better → 50% Enhanced
Week 5+:   If Enhanced Sharpe ≥5% better → 80% Enhanced
Week 7+:   If stable → 100% Enhanced


ROLLBACK PROCEDURE
------------------
If Enhanced underperforms:
1. Automatic rollback reduces allocation to 50% of current
2. Manual rollback: Set enhanced_allocation = 0.05 in config
3. Emergency stop: Set enable_ab_testing = false (legacy only)


PERFORMANCE METRICS TO TRACK
-----------------------------
Primary: Sharpe Ratio (risk-adjusted returns)
Secondary: Win Rate, Profit Factor, Max Drawdown
Tertiary: Accuracy (least important for trading)

Success Criteria:
  ✓ Enhanced Sharpe ≥ Legacy Sharpe + 2%
  ✓ Enhanced Max Drawdown ≤ Legacy Max Drawdown
  ✓ Enhanced Win Rate ≥ Legacy Win Rate - 3%


MONITORING COMMANDS
-------------------
# View current allocation
grep "Enhanced allocation" logs/aether.log | tail -1

# Check auto-rollback events
grep "AUTO-ROLLBACK" logs/aether.log

# View latest report
cat models/ab_test_report.json | python -m json.tool


TROUBLESHOOTING
---------------
Issue: Both models predict NEUTRAL 100%
Fix: This is expected on ranging markets. Check confidence scores instead.

Issue: Auto-rollback triggers immediately
Fix: Increase rollback_threshold to 0.05 (5%) or min_predictions to 200

Issue: No performance difference visible
Fix: Need at least 100+ predictions per model. Wait longer or increase allocation.

Issue: Models fail to load
Fix: Verify model paths, retrain if corrupted: python src/tools/train_phase1b.py


EXPECTED RESULTS
----------------
Based on Phase 1B training:
- Both models achieve ~89% accuracy
- Enhanced has 0.24% better validation loss
- Expected real-world impact: 1-3% Sharpe improvement
- Conservative estimate: Break-even to +2% Sharpe
- Optimistic estimate: +3% to +5% Sharpe


DEPLOYMENT CHECKLIST
--------------------
[ ] Models trained and saved
[ ] config/settings.yaml updated
[ ] src/main_bot.py updated with NexusBrainAB
[ ] Outcome recording added to trade logic
[ ] Monitoring script tested
[ ] Initial allocation set to 20%
[ ] Auto-rollback enabled
[ ] Team briefed on A/B testing
[ ] Rollback procedure documented
[ ] Performance targets defined


NEXT PHASE
----------
After 2-4 weeks of A/B testing:
- If Enhanced performs better → Phase 1C (Larger Architecture)
- If performance similar → Phase 2 (Alternative Approaches)
- If Legacy better → Review feature engineering


CONTACT
-------
For issues or questions about A/B testing deployment:
- Review: DEPLOYMENT_DECISION.md
- Review: PHASE_1B_ANALYSIS.md
- Check logs: logs/aether.log
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_deployment():
    """Validate that all deployment requirements are met"""
    from pathlib import Path
    import yaml
    
    print("Validating Phase 1B A/B Testing Deployment...")
    print("=" * 70)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Legacy model exists
    checks_total += 1
    legacy_path = Path(AB_TESTING_CONFIG["legacy_model_path"])
    if legacy_path.exists():
        print(f"✓ Legacy model found: {legacy_path}")
        checks_passed += 1
    else:
        print(f"✗ Legacy model missing: {legacy_path}")
    
    # Check 2: Enhanced model exists
    checks_total += 1
    enhanced_path = Path(AB_TESTING_CONFIG["enhanced_model_path"])
    if enhanced_path.exists():
        print(f"✓ Enhanced model found: {enhanced_path}")
        checks_passed += 1
    else:
        print(f"✗ Enhanced model missing: {enhanced_path}")
    
    # Check 3: Config file exists
    checks_total += 1
    config_path = Path("config/settings.yaml")
    if config_path.exists():
        print(f"✓ Config file found: {config_path}")
        checks_passed += 1
        
        # Check 4: Config has AB testing settings
        checks_total += 1
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if 'ai_parameters' in config and 'enable_ab_testing' in config['ai_parameters']:
                print(f"✓ AB testing configured in settings.yaml")
                checks_passed += 1
            else:
                print(f"✗ AB testing not configured in settings.yaml")
    else:
        print(f"✗ Config file missing: {config_path}")
        checks_total += 1
    
    # Check 5: NexusBrain AB wrapper exists
    checks_total += 1
    ab_wrapper_path = Path("src/ai_core/nexus_brain_ab.py")
    if ab_wrapper_path.exists():
        print(f"✓ NexusBrain AB wrapper found: {ab_wrapper_path}")
        checks_passed += 1
    else:
        print(f"✗ NexusBrain AB wrapper missing: {ab_wrapper_path}")
    
    # Check 6: AB Testing Manager exists
    checks_total += 1
    ab_manager_path = Path("src/ai_core/ab_testing_manager.py")
    if ab_manager_path.exists():
        print(f"✓ AB Testing Manager found: {ab_manager_path}")
        checks_passed += 1
    else:
        print(f"✗ AB Testing Manager missing: {ab_manager_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print(f"Validation: {checks_passed}/{checks_total} checks passed")
    
    if checks_passed == checks_total:
        print("✓ READY FOR DEPLOYMENT")
        return True
    else:
        print("✗ DEPLOYMENT BLOCKED - Fix issues above")
        return False


def print_deployment_status():
    """Print current deployment status"""
    import json
    from pathlib import Path
    
    report_path = Path(AB_TESTING_CONFIG["report_path"])
    
    if report_path.exists():
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        print("\nCurrent A/B Testing Status:")
        print("=" * 70)
        print(f"Enhanced Allocation: {report.get('enhanced_allocation', 0)*100:.1f}%")
        print(f"Legacy Allocation: {(1-report.get('enhanced_allocation', 0))*100:.1f}%")
        
        if 'models' in report:
            for model_name, stats in report['models'].items():
                print(f"\n{model_name.upper()}:")
                print(f"  Predictions: {stats.get('predictions', 0)}")
                print(f"  Accuracy: {stats.get('accuracy', 0)*100:.2f}%")
                print(f"  Win Rate: {stats.get('win_rate', 0)*100:.2f}%")
                if stats.get('sharpe_ratio'):
                    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.4f}")
                print(f"  Total PnL: ${stats.get('total_pnl', 0):.2f}")
        
        if 'comparison' in report:
            comp = report['comparison']
            print(f"\nCOMPARISON (Enhanced - Legacy):")
            print(f"  Accuracy Diff: {comp.get('accuracy_diff', 0)*100:+.2f}%")
            print(f"  Sharpe Diff: {comp.get('sharpe_diff', 0):+.4f}")
            print(f"  PnL Diff: ${comp.get('pnl_diff', 0):+.2f}")
    else:
        print(f"\nNo A/B testing report found at {report_path}")
        print("Start the bot to begin collecting data.")


if __name__ == "__main__":
    """Run deployment validation"""
    
    print(DEPLOYMENT_INSTRUCTIONS)
    print("\n" + "=" * 70)
    print("RUNNING DEPLOYMENT VALIDATION")
    print("=" * 70 + "\n")
    
    ready = validate_deployment()
    
    if ready:
        print("\n" + "=" * 70)
        print_deployment_status()
        print("\n✓ Deployment validation complete - Ready to deploy!")
    else:
        print("\n✗ Deployment validation failed - Fix issues before deploying")
