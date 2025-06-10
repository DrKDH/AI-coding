**Methodology**
1. Simulation Setup: 300 independent runs per algorithm-environment combination
2. Performance Metrics:
   - Average reward (entire trial period: 0-199)
   - Post-change reward (trials 100-199 for Environment A)
   - Recovery time (trials to reach 90% optimal performance after change)
   - Recovery rate (ratio of achieved to optimal performance)
   - Emotional dynamics (8-dimensional activation patterns for ECIA)
3. Statistical Analysis:
   - Levene's test for variance homogeneity
   - Welch's t-test (for unequal variances) or Student's t-test (for equal variances)
   - Cohen's d for effect size measurement

**Computing Infrastructure**
- Operating System: Compatible with Windows, macOS, and Linux
- Hardware: Standard desktop/laptop (8GB RAM recommended)
- Runtime: ~30-45 minutes for complete experiment suite

**Evaluation Method**
- Comparative Analysis: ECIA compared against three baseline algorithms (Epsilon-Greedy, UCB, Thompson Sampling)
- Environmental Diversity: Performance evaluated across three distinct environment types with varying change patterns
- Statistical Robustness: Recovery rate analysis at multiple sample sizes (10, 100, 300 runs)
- Adaptation Analysis: Recovery time and post-change performance metrics

**Citations**
If you use this code in your research, please cite:
Kim, J., & Kang, D. (2025). Why Evolution Chose Emotions: From Human Emotional Intelligence to Adaptive AI. PeerJ Computer Science. [Under Review]

**License & Contribution Guidelines**
This code is provided for research reproducibility purposes. For any use beyond academic research and reproducibility verification, please contact the authors.

## Contact
For questions about the code or implementation details, please contact:
- Corresponding Author: Daihun Kang (gpk1234567@naver.com)

## Materials & Methods
- Computing infrastructure: Compatible with Windows, macOS, and Linux; Standard desktop/laptop (8GB RAM recommended)
- 3rd party dataset DOI/URL: Not applicable - all environments are synthetically generated within the code
- Data preprocessing steps: Not applicable - rewards are generated in real-time during simulation
- Evaluation method: Comparative analysis using multiple performance metrics (see Methodology section)

See the main manuscript for detailed theoretical background and experimental design. The implementation closely follows the mathematical formulations presented in Sections 2.2.1 and 2.3 of the paper.
