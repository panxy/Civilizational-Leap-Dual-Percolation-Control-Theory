# Civilizational-Leap-Dual-Percolation-Control-Theory
https://img.shields.io/badge/License-MIT-yellow.svg
https://img.shields.io/badge/Status-Active%2520Research-blue
Shang Theory : An operational, open-source framework modeling civilizational dynamics. It applies dual-percolation theory and Bayesian-calibrated thresholds to diagnose transition risks in societies, cities, and DAOs via a quantitative governance dashboard. For researchers in complex systems, computational social science, and governance.
A computable framework modeling societal leap or collapse as a phase transition in competing collaboration networks. From ancient empires to digital DAOs.

Core Proposition: Civilizational outcomes are determined by a race between two percolating networks: a positive-Shang network (ϕ⁺) of collaboration and a negative-Shang network (ϕ⁻) of predation. A leap-forward transition occurs if and only if ϕ⁺ > 0.33 and ϕ⁻ < 0.10.

This project transforms historical narrative into real-time, intervenable systems engineering, providing a quantitative dashboard for the health of societies, organizations, and digital communities.

Table of Contents
Core Concepts

Quick Start

For Researchers & Developers

Case Studies & Validation

Responsible Use Statement

Prediction Registry & Observation

License

Core Concepts
Shang Theory 3.2 defines "Shang" as the endogenous infrastructure for cross-period, codifiable energy-packet transmission that emerges within intelligent groups under survival or prosperity pressures. It models civilizations as a dual-network competitive system:

Network	Driving Mechanism	Macro Metric
Positive-Shang Network (ϕ⁺)	Reciprocity, collaboration, trust. Aims to grow total system energy.	Positive Connectivity ϕ⁺
Negative-Shang Network (ϕ⁻)	Predation, fraud, zero-sum extraction. Aims to redistribute, not create, energy.	Negative Connectivity ϕ⁻
The system's ultimate trajectory is measured by its Transition Potential (TP). The core thresholds are derived via Bayesian calibration on 15 historical and contemporary cases:

Positive Percolation Threshold θ⁺: ϕ⁺ ≥ 0.33

Negative Safety Threshold θ⁻: ϕ⁻ ≤ 0.10

System Transition Threshold: TP ≥ 0.52

The theory is operationalized through seven governing equations that link micro-level agent transmissions to macro-level network phase transitions.

Quick Start
For Analysts & Policymakers: One-Minute Diagnostic
A simplified tool allows you to get a diagnostic by inputting 15 proxy variables.

Clone the repository:

bash
git clone https://github.com/Chongqing-2025/shang-theory.git
cd shang-theory
Provide your data: Open quick_diagnostic.py in a text editor. In the input_data dictionary at the top, replace the 15 example values with data for your city or organization.

Run the diagnostic:

bash
python quick_diagnostic.py
The script will output a full diagnostic report including ϕ⁺, ϕ⁻, TP, and the system state.

For Reproducibility: Full Environment Setup
To reproduce full case studies or develop extensions, use the contained environment:

bash
# Create and activate the Conda environment
conda env create -f environment.yml
conda activate shang-theory-env
🔬 For Researchers & Developers
The src/ directory contains a modular, professional codebase for deep exploration, parameter modification, and extension.

src/core_equations.py: Pure function implementations of the seven core equations.

src/diagnostics.py: The complete pipeline from proxy mapping to final diagnosis.

src/utils.py: Helper functions for data loading and visualization.

Use it as a standard Python library:

python
from src.diagnostics import quick_diagnose, get_default_parameters

# Diagnose with custom parameters
my_params = get_default_parameters()
my_params['omega'] = 4.5  # Adjust the destruction amplifier
result = quick_diagnose([0.044, 0.92, ...], custom_params=my_params)

Case Studies & Validation
The theory's validity is demonstrated through a backtest library of 21 samples, spanning from the Qin Dynasty to 2025 DAOs.

Snapshot of Focus Cases:

Case	ϕ⁺	ϕ⁻	TP	Diagnosis
Chongqing (2024)	0.45	0.06	0.78	✅ Deep Positive Transition
Jakarta (2025)	0.31	0.192	0.19	🚨 Negative-Transition Warning
United States (2024)	0.37	0.12	0.61	⚠️ Fragile Positive Transition
Failed DAOs (Aggregate)	0.27	0.34	-0.78	💀 Negative Transition
Explore the Full Case Library: All raw data, calculation processes, and detailed analysis reports are located in the /case_studies directory. This constitutes the core of the theory's verifiability.



We are looking for: researchers, data scientists, policy analysts, DAO builders, and all who believe in the power of collaborative intelligence.

Responsible Use Statement
SHANG Theory is a complex systems model for analyzing civilizational dynamics and potential transition or collapse risks. It is designed to provide early warning signals and intervention insights, not to serve as a direct basis for political decision-making or value judgments.

All users are expected to adhere to the following core principles:

Scientific Neutrality: This model is a descriptive tool. It does not endorse any specific political, economic, or cultural ideology, nor is it a system for ranking civilizations.

Ethical Primacy: Application of the model must respect human rights and fundamental freedoms. It must not be used to justify oppression or discrimination.

Transparency & Verifiability: We encourage the pre-registration of significant predictions on public channels to uphold the scientific principle of falsifiability.

Prevention of Misuse: Users must not manipulate input data or employ results to promote theories of "civilizational hierarchy."

Disclaimer: The authors and contributors of this project are not legally liable for any consequences arising from the use of the SHANG Theory model. Users are solely responsible for their own analyses and conclusions.

Application Guidelines
1. Academic Research
When citing SHANG Theory in scholarly papers or reports, clearly frame it as a theoretical framework. Avoid overinterpreting its outputs as prescriptive policy advice or moral evaluations.

2. Policy Making
Decision-makers should treat model outputs as one input among many. Always contextualize results with local knowledge and conduct comprehensive social impact assessments. The model is a diagnostic aid—not a substitute for democratic deliberation or ethical judgment.

3. Technical Development
Developers building tools or dashboards based on SHANG Theory must design user interfaces that are clear, accurate, and free from sensationalism. Provide sufficient documentation about the model’s assumptions, limitations, and underlying methodology.

4. Public Communication
When discussing SHANG Theory in media or public forums, accurately convey its role as a scientific instrument. Avoid reductive labels such as “civilization rankings” or deterministic claims that could fuel misunderstanding or polarization.

Disclaimer
While we have made every reasonable effort to ensure the model’s accuracy and robustness, all predictions and analyses involve inherent uncertainty. Therefore, the authors and contributors disclaim all liability for any consequences arising from the use of SHANG Theory or its implementations.

Closing Note
We hope SHANG Theory will serve as a valuable instrument for fostering resilient, adaptive, and flourishing societies worldwide. We also trust that every user will apply it with scientific integrity, ethical awareness, and a commitment to the common good—helping to preserve its neutrality, credibility, and constructive potential.



Prediction Registry & Global Observation
To practice the scientific principles of "transparency" and "falsifiability," we initiate the SHANG Prediction Registry.

1. Submit Your Prediction
We encourage researchers worldwide to make quantifiable predictions based on the theory and register them before the predicted event.

How: Create a new Issue on this repository's GitHub Issues page using the [Prediction] label.

Requirements: Clearly describe the event, time window, specific model metric thresholds (e.g., "ϕ⁻ will exceed 0.18 by Q2 2026"), and intended data sources for verification.

2. Quarterly Civilizational Health Snapshot
The project will regularly (quarterly) publish a Global TP & Connectivity Report, updating the diagnostic status of major entities and reviewing registered predictions.

Goal: To establish a continuously observable, publicly auditable record, transforming theoretical development into an open, cumulative, global collaborative process.


How to Cite
If Shang Theory informs your work, please cite it as follows:

bibtex
@software{shang_theory_2025,
  author = {Chongqing·2025},
  title = {Shang Theory: A Dual-Percolation Control Theory of Civilizational Transition},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Chongqing-2025/shang-theory}},
  version = {3.2}
}

📜 License
This transformative work is open to the world under the MIT License. See the LICENSE file for details.

Theory. Measured. Engineered. — This is more than a model; it's an invitation to participate in the responsible understanding and steering of our complex systems.
