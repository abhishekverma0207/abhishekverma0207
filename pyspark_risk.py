#Importing required libraries
import warnings
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
from sklearn.decomposition import PCA as pcap
from sklearn.decomposition import KernelPCA as kpca
import numpy as np
import pandas as pd
from pyspark.sql.functions import col, udf, pandas_udf, PandasUDFType, array, struct , collect_list, lit, when
from pyspark.sql.types import IntegerType, DoubleType, StringType, FloatType, LongType, ArrayType
from pyspark.sql import SparkSession
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
warnings.filterwarnings('ignore')

#Creating logger
logger = logging.getLogger('clf_risk_log')

#Class for applying the classification algorithm and setting the risks
class CLF_RISK(object):

    #__init__ for initialising
    def __init__(self):
        self.spark =  SparkSession\
            .builder\
            .enableHiveSupport()\
            .appName("PREP")\
            .config('spark.driver.memory', '20G')\
            .config('spark.driver.memoryOverhead', '7G')\
            .config('spark.executor.memory', '4G')\
            .config('spark.executor.instances', '100')\
            .config('spark.executor.cores', '5')\
            .config('spark.executor.memoryOverhead', '28G')\
            .config('spark.sql.debug.maxToStringFields', '1000')\
            .config('spark.sql.parquet.writeLegacyFormat', 'true')\
            .config('spark.python.worker.memory', '10G')\
            .config('spark.debug.maxToStringFields', '200')\
            .getOrCreate()
        logger.info("Code instantiated")

#clf_model contains the classification model
    def clf_model(df_f):
        pca_model = pcap(.97)
        planogram_id = [i.planogram_id for i in df_f]
        dpci = [i.dpci for i in df_f]
        dept_cls = [i.dept_cls for i in df_f]
        Blended_score = [i.Blended_score for i in df_f]
        mdse_dept_ref_i = [i.mdse_dept_ref_i for i in df_f]
        store_counts = [i.store_counts for i in df_f]
        target = [i.Blended_score for i in df_f]
        X_df = []
        [X_df.append([i.planogram_height, i.planogram_width, i.planogram_depth, i.fixture_depth, i.fixture_width, \
            i.fixture_height, i.position_item_height, i.position_item_width, i.position_item_depth, \
            i.position_horizontal_facing, i.position_vertical_facing, i.position_depth_facing, \
            i.Position_max_per_facings, i.position_capacity, i.plan_item_d_ratio, i.fix_item_d_ratio]) for i in df_f]
        X_train = X_df
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        pca_model.fit(X_train)
        xgb_cl = xgb.XGBClassifier()
        model = xgb_cl.fit(pca_model.transform(X_train), target)
        score = model.predict(pca_model.transform(scaler.fit_transform(X_df)))
        score_2 = model.predict_proba(pca_model.transform(scaler.fit_transform(X_df)))[:,1]
        preds = [float("{:.6f}".format(x)) for x in score]
        probs = [float("{:.6f}".format(x)) for x in score_2]
        return zip(planogram_id, dpci, dept_cls, Blended_score, mdse_dept_ref_i, store_counts, preds, probs)

    #clf_risk gets the data, executes the classification model and calculates risks
    def clf_risk_data(self):
        df = self.query_info.select(
            col("planogram_id")
            ,col("dpci")
            ,col("dept_cls")
            ,col("Blended_score")
            ,col("mdse_dept_ref_i")
            ,col("store_counts")
            ,col("planogram_height")
            ,col("planogram_width")
            ,col("planogram_depth")
            ,col("fixture_depth")
            ,col("fixture_width")
            ,col("fixture_height")
            ,col("position_item_height")
            ,col("position_item_width")
            ,col("position_item_depth")
            ,col("position_horizontal_facing")
            ,col("position_vertical_facing")
            ,col("position_depth_facing")        
            ,col("Position_max_per_facings")
            ,col("position_capacity")
            ,col("plan_item_d_ratio")        
            ,col("fix_item_d_ratio")
        )
        df_rdd = df.rdd.map(lambda row: ((row.mdse_dept_ref_i), row))
        df_rdd_grp = df_rdd.groupByKey().mapValues(list)
        df_rdd_run = df_rdd_grp.map(lambda x : clf_model(x[1]))
        df_rdd_run_final = df_rdd_run.flatMap(lambda x: x)
        Schema_data=StructType([StructField("planogram_id", StringType()),\
        StructField("dpci",StringType()),\
        StructField("dept_cls",StringType()),\
        StructField("Blended_score",IntegerType()),\
        StructField("mdse_dept_ref_i",StringType()),\
        StructField("store_counts",IntegerType()),\
        StructField("predictions",FloatType()),\
        StructField("probability",FloatType())])
        final = spark.createDataFrame(df_rdd_run_final, schema=Schema_data)
        complete = data.join(final, ['planogram_id', 'dpci', 'mdse_dept_ref_i', 'dept_cls', 'Blended_score', 'store_counts'], 'left')\
            .drop(final.planogram_id).drop(final.dpci).drop(final.mdse_dept_ref_i).drop(final.dept_cls).drop(final.Blended_score)\
            .drop(final.store_counts)
        data0 = complete.where(complete.Blended_score == 1)
        data1 = data0.withColumn("risk_iso", when(data0.avg_scaled_iso <= 0.5, '5_Very_Low')\
            .when((data0.avg_scaled_iso > 0.5) & (data0.avg_scaled_iso <= 0.7), '4_Low')\
            .when((data0.avg_scaled_iso > 0.7) & (data0.avg_scaled_iso <= 0.9), '3_Medium')\
            .when((data0.avg_scaled_iso > 0.9) & (data0.avg_scaled_iso <= 0.95), '2_High')\
                .otherwise('1_Very_High'))
        data2 = data1.withColumn("risk_svm", when(data1.avg_scaled_svm <= 0.5, '5_Very_Low')\
            .when((data1.avg_scaled_svm > 0.5) & (data1.avg_scaled_svm <= 0.7), '4_Low')\
            .when((data1.avg_scaled_svm > 0.7) & (data1.avg_scaled_svm <= 0.9), '3_Medium')\
            .when((data1.avg_scaled_svm > 0.9) & (data1.avg_scaled_svm <= 0.95), '2_High')\
            .otherwise('1_Very_High'))
        data3 = data2.withColumn("risk_clf", when(data2.probability <= 0.5, '5_Very_Low')\
            .when((data2.probability > 0.5) & (data2.probability <= 0.7), '4_Low')\
            .when((data2.probability > 0.7) & (data2.probability <= 0.9), '3_Medium')\
            .when((data2.probability > 0.9) & (data2.probability <= 0.95), '2_High')\
            .otherwise('1_Very_High'))
        data4 =  data3.withColumn("weighted_scaled", 0.6*data3['avg_scaled_iso'] + 0.2*data3['avg_scaled_svm'] + \
            0.2*data3['probability'])
        output = data4.withColumn("risk_weighted", when(data4.weighted_scaled <= 0.5, '5_Very_Low')\
            .when((data4.weighted_scaled > 0.5) & (data4.weighted_scaled <= 0.7), '4_Low')\
            .when((data4.weighted_scaled > 0.7) & (data4.weighted_scaled <= 0.9), '3_Medium')\
            .when((data4.weighted_scaled > 0.9) & (data4.weighted_scaled <= 0.95), '2_High')\
            .otherwise('1_Very_High'))
        output = output.withColumn("risk_weighted", F.when(((col('position_item_height') == 1) & \
            (col("position_item_width") == 1) & (col("position_item_depth") == 1)), '1_Very_High').otherwise(output.risk_weighted))
        output = output.withColumn("main_cause", F.when(((col('position_item_height') == 1) & \
            (col("position_item_width") == 1) & (col("position_item_depth") == 1)), 'item_hvd_111').otherwise(output.main_cause))
        output = output.where(col('placement_set_date') <= F.date_add(col('greg_d'), 90))
        output = output.where(col('placement_set_date') >= col('greg_d'))
        self.spark.sql('drop table if exists prd_atc_fnd.atcad_risk_out')
        output.write.partitionBy('greg_d', 'placement_set_date').\
            mode('overwrite').format('ORC').saveAsTable('prd_atc_fnd.atcad_risk_out')
        logger.info("Data has been processed by applying classification model and risks are also allotted")

    #query_data gets the data from stg_sc_atcad.oozie_output_results1
    def query_data(self):
        query = """
            select 
                * 
            from stg_sc_atcad.oozie_output_results1"""
        self.query_info = self.spark.sql(query)
        logger.info("Query obtained")

    #startCLFRISK starts the code for getting risks after applying classification algorithm on data
    def startCLFRISK(self):
        self.query_data()
        self.clf_risk_data()

#Code to run to get data
pull = CLF_RISK()
pull.startCLFRISK()
