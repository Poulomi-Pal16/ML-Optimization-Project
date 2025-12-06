Project Title: Optimization of Artificial Neural Networks
Student Name: Poulomi Pal

------------------------------------------------------------
📘 Project Description
------------------------------------------------------------
This project focuses on optimizing Artificial Neural Networks (ANNs) using three metaheuristic algorithms — 
Teaching Learning-Based Optimization (TLBO), Spotted Hyena Optimizer (SHO), and Harris Hawk Optimizer (HHO) — 
to enhance prediction accuracy and model performance.

The results are compared using evaluation metrics such as R², MSE, and RMSE.

------------------------------------------------------------
📂 Folder Structure
------------------------------------------------------------
24bce8026_poulomipal/
│
├── ML/
│   ├── comparison_plot.py   → Generates a bar chart comparing R², MSE, and RMSE across models.
│   ├── tlbo_ann.py          → Implements the TLBO-ANN optimization and training.
│   ├── sho_ann.py           → Implements the SHO-ANN optimization and training.
│   ├── hho_mlp.py           → Implements the HHO-MLP optimization and training.
│   ├── Data.csv             → Dataset used for training and evaluation.
│   └── .vscode/             → VS Code configuration files.
│
└── Project_Report.docx       → Detailed project report containing aim, methodology, code overview, results, and discussion.

------------------------------------------------------------
⚙️ How to Run the Code
------------------------------------------------------------
1. Open the ML folder in VS Code.
2. Ensure all required Python libraries are installed:
   pip install numpy pandas scikit-learn matplotlib
3. Run each model script individually:
   python tlbo_ann.py
   python sho_ann.py
   python hho_mlp.py
4. After execution, run:
   python comparison_plot.py
   This will generate a bar chart comparing all models.

------------------------------------------------------------
📊 Output
------------------------------------------------------------
- Console output: R², MSE, and RMSE values for each model.
- Comparison plot: Displays visual performance differences across the three models.
