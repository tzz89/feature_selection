#! /bin/bash

# downloading housing_price regression data from kaggle
DATA_DIR="data" 
HOUSING_GDRIVE_URL="https://docs.google.com/uc?export=download&id=1m59q5DfAmbFYBHUNSGIdEjTCe_ytaeHg"
HOUSING_FP="${DATA_DIR}/housing_price_regression"
HOUSING_ZIP_FP="${DATA_DIR}/housing_price_regression.zip"


curl -L ${HOUSING_GDRIVE_URL} -o ${HOUSING_ZIP_FP}
unzip ${HOUSING_ZIP_FP} -d ${HOUSING_FP} 
rm ${HOUSING_ZIP_FP}