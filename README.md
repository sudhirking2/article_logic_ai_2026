```
logic-aware-extraction/
│
├── README.md
├── requirements.txt
├── setup.py
│
├── logify/                     # Text → Logified Structure
│   ├── init.py
│   ├── propositions.py         # Extract propositions
│   ├── constraints.py          # Extract hard/soft constraints
│   ├── weights.py              # Assign weights
│   ├── schema.py               # Build dictionary
│   └── update.py               # Add new text/propositions/constraints
│
├── logic_solver/               # All logic calculations
│   ├── init.py
│   ├── encoding.py             # Propositions → Logic software (e.g., Max-SAT)
│   └── maxsat.py               # Solver interface
│
├── interface/                  # User-LLM interactions
│   ├── init.py
│   ├── translate.py            # NL query → formal
│   ├── interpret.py            # Solver output → NL
│   └── refine.py               # Self-refinement
│
├── outputs/
│   ├── logified/
│   └── results/
│
└── experiments/
```
