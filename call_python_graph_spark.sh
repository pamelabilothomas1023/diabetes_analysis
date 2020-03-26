!bin/bash

export PATH=${PATH}:/opt/spark/bin
export PYSPARK_PYTHON=python3.5

spark-submit read_diagnosis_data.py '584_DIA 586_DIA 585_DIA 403_DIA 404_DIA 581_DIA 583_DIA 588_DIA N18_DIA N17_DIA N19_DIA I12_DIA I13_DIA N04_DIA N05_DIA N08_DIA N25_DIA 593_DIA' 'KID_DIA'
spark-submit read_demographic_file.py 'KID_DIA'
spark-submit read_clin_var_file.py 'KID_DIA'
spark-submit read_clin_var_file_status.py 'KID_DIA'
spark-submit Create_DAG_Spark.py 'KID_DIA'
spark-submit readpythongraph_spark.py 'KID_DIA'
spark-submit dataanalysis_spark.py 'KID_DIA' > 'KID_DIA_spark.txt'
spark-submit predict_spark1.py 'KID_DIA' 7

spark-submit read_diagnosis_data.py '571_DIA 572_DIA 573_DIA K76_DIA K75_DIA' 'LIV_DIA'
spark-submit read_demographic_file.py 'LIV_DIA'
spark-submit read_clin_var_file.py 'LIV_DIA'
spark-submit read_clin_var_file_status.py 'LIV_DIA'
spark-submit Create_DAG_Spark.py 'LIV_DIA'
spark-submit readpythongraph_spark.py 'LIV_DIA'
spark-submit dataanalysis_spark.py 'LIV_DIA' > 'LIV_DIA_spark.txt'
spark-submit predict_spark1.py 'LIV_DIA' 7

spark-submit read_diagnosis_data.py '428_DIA I50_DIA' 'HFL_DIA'
spark-submit read_demographic_file.py 'HFL_DIA'
spark-submit read_clin_var_file.py 'HFL_DIA'
spark-submit read_clin_var_file_status.py 'HFL_DIA'
spark-submit Create_DAG_Spark.py 'HFL_DIA'
spark-submit readpythongraph_spark.py 'HFL_DIA'
spark-submit dataanalysis_spark.py 'HFL_DIA' > 'HFL_DIA_spark.txt'
spark-submit predict_spark1.py 'HFL_DIA' 7

spark-submit read_diagnosis_data.py '410_DIA I21_DIA 412_DIA' 'HAT_DIA'
spark-submit read_demographic_file.py 'HAT_DIA'
spark-submit read_clin_var_file.py 'HAT_DIA'
spark-submit read_clin_var_file_status.py 'HAT_DIA'
spark-submit Create_DAG_Spark.py 'HAT_DIA'
spark-submit readpythongraph_spark.py 'HAT_DIA'
spark-submit dataanalysis_spark.py 'HAT_DIA' > 'HAT_DIA_spark.txt'
spark-submit predict_spark1.py 'HAT_DIA' 7

spark-submit read_diagnosis_data.py '435_DIA G45_DIA 430_DIA 431_DIA I60_DIA I61_DIA 432_DIA I62_DIA 436_DIA 433_DIA 434_DIA' 'STR_DIA'
spark-submit read_demographic_file.py 'STR_DIA'
spark-submit read_clin_var_file.py 'STR_DIA'
spark-submit read_clin_var_file_status.py 'STR_DIA'
spark-submit Create_DAG_Spark.py 'STR_DIA'
spark-submit readpythongraph_spark.py 'STR_DIA'
spark-submit dataanalysis_spark.py 'STR_DIA' > 'STR_DIA_spark.txt'
spark-submit predict_spark1.py 'STR_DIA' 7

spark-submit read_diagnosis_data.py '362_DIA H35_DIA' 'RET_DIA'
spark-submit read_demographic_file.py 'RET_DIA'
spark-submit read_clin_var_file.py 'RET_DIA'
spark-submit read_clin_var_file_status.py 'RET_DIA'
spark-submit Create_DAG_Spark.py 'RET_DIA'
spark-submit readpythongraph_spark.py 'RET_DIA'
spark-submit dataanalysis_spark.py 'RET_DIA' > 'RET_DIA_spark.txt'
spark-submit predict_spark1.py 'RET_DIA' 7
