# Datasets

The three benchmark datasets are public and are **not redistributed** here. Download them from the
original sources and place the files exactly as below; the loaders in `code/data/` read these paths.

```
code/data/
├── NSL_KDD_repo/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── UNSW_NB15/
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
└── CICIDS2017/                      (the 8 MachineLearningCVE CSV files)
    ├── Monday-WorkingHours.pcap_ISCX.csv
    ├── Tuesday-WorkingHours.pcap_ISCX.csv
    ├── Wednesday-workingHours.pcap_ISCX.csv
    ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
    ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
    └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

Sources:
- NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html
- UNSW-NB15 (official training/testing partition): https://research.unsw.edu.au/projects/unsw-nb15-dataset
- CICIDS2017 (MachineLearningCVE CSVs): https://www.unb.ca/cic/datasets/ids-2017.html

You only need the raw datasets to retrain the classifiers. Every arbitration, feedback, UGAA and
archetype experiment can be rerun from the prediction files in the results archive (see README.md).
