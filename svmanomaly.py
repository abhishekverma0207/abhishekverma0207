#Importing required libraries
from pyspark.sql import Window
from pyspark.sql.functions import col, count, rand, collect_list, explode, struct, count, lit, pow, sum, abs, when
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.functions import date_add, monotonically_increasing_id
from pyspark.sql.functions import *
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.sql import HiveContext, SparkSession, SQLContext
from pyspark import SparkContext
from datetime import timedelta
from datetime import date
from datetime import datetime as dt
from pyspark.sql import functions as F
from pyspark.sql.functions import current_date, datediff
from pyspark.sql.functions import regexp_extract
import pyspark.sql.types as pst
from pyspark.sql.types import FloatType
import math
import pandas as pd
import numpy as np
from pyspark.sql.types import *
from sklearn.preprocessing import StandardScaler
from scipy import stats as st
from sklearn.decomposition import PCA as pcap
from sklearn.decomposition import KernelPCA as kpca
from sklearn import svm
from sklearn.ensemble import IsolationForest
from pyspark.sql import DataFrameStatFunctions as statFunc
warnings.filterwarnings('ignore')

#Creating logger
logger = logging.getLogger('iso_svm_log')

#Class for running the iso_svm algorithm on the data
class ISO_SVM(object):

    #__init__ for initialising
    def __init__(self):
        self.spark =  SparkSession\
            .builder\
            .enableHiveSupport()\
            .appName("ISO_SVM")\
            .config('spark.driver.memory', '20G')\
            .config('spark.driver.memoryOverhead', '7G')\
            .config('spark.executor.memory', '4G')\
            .config('spark.executor.instances', '100')\
            .config('spark.executor.cores', '5')\
            .config('spark.executor.memoryOverhead', '28G')\
            .config('spark.sql.debug.maxToStringFields', '1000')\
            .config('spark.sql.parquet.writeLegacyFormat', 'true')\
            .getOrCreate()
        logger.info("Code instantiated")

    #loc_dept_cls_f function
    def loc_dept_cls_f(loc, dpt):
        return str(loc) + "_" + str(dpt)
    loc_dept_cls_udf = udf(loc_dept_cls_f)

    #loc_dpci function
    def loc_dpci(loc, dpci):
        return str(loc) + "_" + str(dpci)
    loc_dpci_udf = udf(loc_dpci)

    #loc_dept_cls_sub function
    def loc_dept_cls_sub(loc, dpt_cls_sub):
        return str(loc) + "_" + str(dpt_cls_sub)
    loc_dept_cls_sub_udf = udf(loc_dept_cls_sub)

    #loc_ft function
    def loc_ft(loc, ft):
        return str(loc) + "_" + str(ft)
    loc_ft_udf = udf(loc_ft)

    #loc_deptcls_mch function
    def loc_deptcls_mch(loc, dpt, merch):
        return str(loc) + "_" + str(dpt) + "_" + str(merch)
    loc_deptcls_mch_udf = udf(loc_deptcls_mch)

    #z_score_w function
    def z_score_w(col, w):
        avg_ = F.mean(col).over(w)
        sd_ = F.stddev(col).over(w)
        return (col - avg_) / sd_
    z_score_w_udf = udf(z_score_w, FloatType())

    #z_score_test function
    def z_score_test(col):
        return st.zscore(col)
    z_score_test_udf = udf(z_score_test, FloatType())

    #z_score_w2 function      
    def z_score_w2(col, w):
        avg_ = F.mean(col).over(w)
        sd_ = F.stddev(col).over(w)
        outr = avg_ + 1.5*sd_
        col2 = F.when(col >= outr, outr).otherwise(col)
        avg2_ = F.mean(col2).over(w)
        sd2_ = F.stddev(col2).over(w)
        return (col - avg2_) / sd2_
    z_score_w2_udf = udf(z_score_w2, FloatType())

    #median_val function
    def median_val(x):
        return float(np.median(x))
    median_udf = udf(median_val, FloatType())

    #min_val function
    def min_val(x):
        return float(min(x))
    min_udf = udf(min_val, FloatType())

    #iso_svm contains the main iso-svm algorithm
    def iso_svm(atc_fctrs):
        pca_model = pcap(.97)
        co_loc_ref_i = [i.co_loc_ref_i for i in atc_fctrs]
        dept_class = [i.dept_class for i in atc_fctrs]
        dpci = [i.dpci for i in atc_fctrs]
        loc_dept_cls = [i.loc_dept_cls for i in atc_fctrs]
        loc_deptcls_mch = [i.loc_deptcls_mch for i in atc_fctrs]
        planogram_id = [i.planogram_id for i in atc_fctrs]
        X_df=[]
        [X_df.append([i.atc_value, i.zscore_median, i.zscore_lt3, i.planogram_depth, i.position_depth_facing, \
            i.atc_zscore_dptcls, i.atc_zscore_locdptcls, i.atc2_zscore_locdptcls_mch, i.d_atc_zscore_dptcls, i.d_zscore_mean]) for i in atc_fctrs]
        x_train = X_df
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        pca_model.fit(x_train)
        anomaly_algorithms = IsolationForest(
            contamination=0.25, random_state=42, n_jobs=-1, bootstrap=True)
        model = anomaly_algorithms.fit(pca_model.transform(x_train))
        score = model.predict(pca_model.transform(scaler.fit_transform(X_df)))
        iso_score = model.decision_function(pca_model.transform(scaler.fit_transform(X_df)))
        score_a_1 = [float("{:.6f}".format(x)) for x in score]
        score_a_2 = [float("{:.6f}".format(x)) for x in iso_score]
        one_class_svm = svm.OneClassSVM(nu=0.25, kernel="rbf", gamma=.5)
        model = one_class_svm.fit(pca_model.transform(x_train))
        anomaly_prediction = model.predict(pca_model.transform(x_train))
        anomaly_score = model.decision_function(pca_model.transform(x_train))
        anomaly_prediction_1 = [float("{:.6f}".format(x)) for x in anomaly_prediction]
        anomaly_prediction_2 = [float("{:.6f}".format(float(x))) for x in anomaly_score]
        return zip(co_loc_ref_i,dept_class,dpci,loc_dept_cls,loc_deptcls_mch,planogram_id,score_a_1,score_a_2,\
            anomaly_prediction_1,anomaly_prediction_2)

    #iso_svm_data applies the iso_svm algorithm on the data
    def iso_svm_data(self):
        atc_data = self.query_info.dropDuplicates(['co_loc_ref_i', 'greg_d', 'placement_set_date', 'dpci', 'planogram_id', 'dept_class', 'mdse_dept_ref_i', \
            'mdse_dept_n', 'mdse_clas_ref_i', 'mdse_clas_n', 'item_description', 'subclass_name', 'subclass_id', 'planogram_description', \
            'position_merch_style', 'planogram_type', 'display_type', 'fixture_type'])
        atc_data_grp = atc_data.groupby('loc_dept_cls').agg(expr('percentile(atc_value, array(0.5))')[0]\
            .alias('atc_median_locdptcls'))
        atc_data_cmp = atc_data.join(atc_data_grp, atc_data.loc_dept_cls == atc_data_grp.loc_dept_cls,'left')\
            .drop(atc_data_grp.loc_dept_cls)
        w = Window.partitionBy("subclass_id")
        pre_df = atc_data_cmp.withColumn("atc_mean_sub", F.avg("atc_value").over(w))\
            .withColumn("atc_stdev_sub", F.stddev("atc_value").over(w))
        pre_df_1 = pre_df.withColumn("atc_zscore_sub",(pre_df.atc_value - pre_df.atc_mean_sub)/ pre_df.atc_stdev_sub)\
            .withColumn("outr",(pre_df.atc_mean_sub+ 1.5*pre_df.atc_stdev_sub))
        pre_df_2 = pre_df_1.withColumn("col2", F.when(pre_df_1.atc_value > pre_df_1.outr, pre_df_1.outr)\
            .otherwise(pre_df_1.atc_value))
        pre_df_3 = pre_df_2.withColumn( "avg2_",F.avg("col2").over(w)).withColumn( "std2_",F.stddev("col2").over(w))
        pre_df_4 = pre_df_3.withColumn( "atc2_zscore_sub",(pre_df_3.atc_value - pre_df_3.avg2_)/ pre_df_3.std2_)\
            .drop(pre_df_3.outr).drop(pre_df_3.col2).drop(pre_df_3.avg2_).drop(pre_df_3.std2_)
        w = Window.partitionBy("dept_class")
        pre_df_4_grp = pre_df_4.groupby('dept_class').agg(expr('percentile(atc_value, array(0.5))')[0]\
            .alias('atc_median_dptcls'))
        pre_df_4_cmp = pre_df_4.join(pre_df_4_grp,['dept_class'],'left')
        pre_df_5 = pre_df_4_cmp.withColumn("atc_mean_dptcls", F.avg("atc_value").over(w)).withColumn("atc_stdev_dptcls",\
            F.stddev("atc_value").over(w))
        pre_df_6 = pre_df_5.withColumn("atc_zscore_dptcls",(pre_df_5.atc_value - pre_df_5.atc_mean_dptcls)/ pre_df_5\
            .atc_stdev_dptcls)
        pre_df_6_temp = pre_df_6.withColumn("atc_zscore_dptcls", when(pre_df_6.atc_stdev_dptcls == 0, \
                pre_df_6.atc_value - pre_df_6.atc_mean_dptcls).otherwise(pre_df_6.atc_zscore_dptcls))
        pre_df_6_temp2 = pre_df_6_temp.withColumn("atc_zscore_dptcls_NULL", when(pre_df_6_temp.atc_stdev_dptcls ==\
                0, lit(1)).otherwise(lit(0)))
        w = Window.partitionBy("loc_dept_cls")
        pre_df_7 = pre_df_6_temp2.withColumn( "atc_mean_locdptcls", F.avg("atc_value").over(w)).withColumn( "atc_stdev_locdptcls",\
            F.stddev("atc_value").over(w))
        pre_df_8 = pre_df_7.withColumn("atc_zscore_locdptcls",(pre_df_7.atc_value - pre_df_7.atc_mean_locdptcls)/ \
            pre_df_7.atc_stdev_locdptcls)
        pre_df_8_temp = pre_df_8.withColumn("atc_zscore_locdptcls", when(pre_df_8.atc_stdev_locdptcls == 0, \
                pre_df_8.atc_value - pre_df_8.atc_mean_locdptcls).otherwise(pre_df_8.atc_zscore_locdptcls))
        pre_df_8_temp2 = pre_df_8_temp.withColumn("atc_zscore_locdptcls_NULL", when(pre_df_8_temp.atc_stdev_locdptcls ==\
                0, lit(1)).otherwise(lit(0)))
        w = Window.partitionBy("loc_deptcls_mch")
        pre_df_9 = pre_df_8_temp2.withColumn("atc_mean_locdptcls_mch", F.avg("atc_value").over(w))\
            .withColumn("atc_stdev_locdptcls_mch", F.stddev("atc_value").over(w))
        pre_df_10 = pre_df_9.withColumn("atc_zscore_locdptcls_mch",(pre_df_9.atc_value - pre_df_9.atc_mean_locdptcls_mch)\
            / pre_df_9.atc_stdev_locdptcls_mch).withColumn("outr",(pre_df_9.atc_mean_locdptcls_mch+ \
            1.5 * pre_df_9.atc_stdev_locdptcls_mch))
        pre_df_10_temp = pre_df_10.withColumn("atc_zscore_locdptcls_mch", when(pre_df_10.atc_stdev_locdptcls_mch == 0, \
                pre_df_10.atc_value - pre_df_10.atc_mean_locdptcls_mch).otherwise(pre_df_10.atc_zscore_locdptcls_mch))
        pre_df_10_temp2 = pre_df_10_temp.withColumn("atc_zscore_locdptcls_mch_NULL", when(pre_df_10_temp.atc_stdev_locdptcls_mch ==\
                0, lit(1)).otherwise(lit(0)))
        pre_df_11 = pre_df_10_temp2.withColumn("col2", F.when(pre_df_10_temp2.atc_value > pre_df_10_temp2.outr, pre_df_10_temp2.outr)\
            .otherwise(pre_df_10_temp2.atc_value))
        pre_df_12 = pre_df_11.withColumn( "avg2_",F.avg("col2").over(w)).withColumn( "std2_",F.stddev("col2").over(w))
        pre_df_13 = pre_df_12.withColumn( "atc2_zscore_locdptcls_mch",(pre_df_12.atc_value - pre_df_12.avg2_)/ \
            pre_df_12.std2_).drop(pre_df_12.outr).drop(pre_df_12.col2).drop(pre_df_12.avg2_).drop(pre_df_12.std2_)\
            .withColumn('d_zscore_mean', F.when(abs(col('zscore_mean')) > 2.8, 1).otherwise(0))\
            .withColumn('d_atc_zscore_dptcls', F.when(abs(col('atc_zscore_dptcls')) > 2.8, 1).otherwise(0))\
            .withColumn('coef_var', (col('atc_stdev_locdptcls')/col('atc_mean_locdptcls'))*100)\
            .withColumn('coef_var_sub', (col('atc_stdev_sub')/col('atc_mean_sub'))*100)
        df2 = pre_df_13.repartition(3000, "loc_dept_cls")
        df3 = df2.drop('display_type').dropDuplicates(['greg_d', 'placement_set_date', 'co_loc_ref_i', 'dpci', 'planogram_id', 'dept_class',\
                'mdse_dept_ref_i', 'mdse_dept_n', 'mdse_clas_ref_i', 'mdse_clas_n', 'planogram_description', 'subclass_name', 'subclass_id',\
                'position_merch_style', 'planogram_type', 'fixture_type'])
        df4 = df3.select(
            col('co_loc_ref_i'),
            col('dpci'),
            col('planogram_id'),
            col('loc_dept_cls'),
            col('dept_class'),
            col('loc_deptcls_mch'),
            col('subclass_id'),
            col('atc_value'),
            col('zscore_median'),
            col('zscore_lt3'),
            col('planogram_depth'),
            col('position_depth_facing'),
            col('atc_zscore_dptcls'),
            col('atc_zscore_locdptcls'),
            col("atc_zscore_dptcls_NULL"),
            col("atc_zscore_locdptcls_NULL"),
            col("atc_zscore_locdptcls_mch_NULL"),
            col('atc2_zscore_locdptcls_mch'),
            col('d_atc_zscore_dptcls'),
            col('d_zscore_mean'))
        df5 = df4.na.drop()
        df_atc_model_rdd = df5.rdd.map(lambda row: ((row.loc_dept_cls), row))
        df_atc_model_rdd_grp = df_atc_model_rdd.groupByKey().mapValues(list)
        df_atc_model_run = df_atc_model_rdd_grp.map(lambda x:iso_svm(x[1]))
        df_atc_model_run_1 = df_atc_model_run.flatMap(lambda x: x)
        Schema_data = StructType([StructField("co_loc_ref_i", StringType()),\
        StructField("dept_class", StringType()),\
        StructField("dpci", StringType()),\
        StructField("loc_dept_cls", StringType()),\
        StructField("loc_deptcls_mch", StringType()),\
        StructField("planogram_id", StringType()),\
        StructField("results_iso", FloatType()),\
        StructField("results_iso2", FloatType()),\
        StructField("results_svm", FloatType()),\
        StructField("results_svm2", FloatType())])
        atc_iso_svm_df = spark.createDataFrame(df_atc_model_run_1,schema=Schema_data)
        atc_iso_svm_df.persist()
        post_df_14 = pre_df_13.join(atc_iso_svm_df,['co_loc_ref_i','dpci','planogram_id'], 'left')\
            .drop(atc_iso_svm_df.dept_class).drop(atc_iso_svm_df.loc_dept_cls).drop(atc_iso_svm_df.loc_deptcls_mch)
        iso_std = post_df_14.agg(F.stddev("results_iso2")).collect()[0][0]
        svm_std = post_df_14.agg(F.stddev("results_svm2")).collect()[0][0]
        new1 = post_df_14.withColumn('scaled_iso', (col('results_iso2') - 0)/(iso_std))
        new2 = new1.withColumn('scaled_iso', F.when(col('scaled_iso') > 3.2, 3.2).otherwise(col('scaled_iso')))
        new3 = new2.withColumn('scaled_iso', F.when(col('scaled_iso') < -3.2, -3.2).otherwise(col('scaled_iso')))
        new4 = new3.withColumn('scaled_iso', 1 + F.exp(col('scaled_iso')))
        new5 = new4.withColumn("scaled_iso", F.pow(col('scaled_iso'), lit(-1)))
        new6 = new5.withColumn('scaled_svm', (col('results_svm2') - 0)/(svm_std))
        new7 = new6.withColumn('scaled_svm', F.when(col('scaled_svm') > 3.2, 3.2).otherwise(col('scaled_svm')))
        new8 = new7.withColumn('scaled_svm', F.when(col('scaled_svm') < -3.2, -3.2).otherwise(col('scaled_svm')))
        new9 = new8.withColumn('scaled_svm', 1 + F.exp(col('scaled_svm')))
        new10 = new9.withColumn("scaled_svm", F.pow(col('scaled_svm'), lit(-1)))
        new11 = new10.withColumn('prediction_iso', F.when(col('results_iso') <= 0, 1).otherwise(0))
        new12 = new11.withColumn('prediction_svm', F.when(col('results_svm') <= 0, 1).otherwise(0))
        post_df_15 = new12.withColumn('Blended_score', F.when(col('coef_var') <= 75, col('prediction_svm'))
                            .otherwise(col('prediction_iso')))
        atc_anomaly_out = post_df_15.select(
            col("greg_d")
            ,col("placement_set_date")
            ,col("co_loc_ref_i")
            ,col("dpci")
            ,col("ipc_item")
            ,col('planogram_id')
            ,col("atc_value")
            ,col('prediction_svm')
            ,col("prediction_iso")
            ,col('Blended_score')
            ,col("dept_class")
            ,col("mdse_dept_ref_i")
            ,col("mdse_dept_n")
            ,col("mdse_clas_ref_i")
            ,col("mdse_clas_n")
            ,col("subclass_name")
            ,col('subclass_id')
            ,col("item_description")
            ,col("planogram_description")
            ,col("position_merch_style")
            ,col("planogram_type")
            ,col("fixture_type")
            ,col("planogram_height")
            ,col("planogram_width")
            ,col("planogram_depth")
            ,col('fixture_depth')
            ,col('fixture_width')
            ,col('fixture_height')
            ,col('position_item_height')
            ,col('position_item_width')
            ,col('position_item_depth' )
            ,col("position_horizontal_facing")
            ,col("position_vertical_facing")
            ,col("position_depth_facing")
            ,col('Position_max_per_facings')
            ,col('position_hvd_111')
            ,col("cals_median_position_capacity")
            ,col("cals_mean_position_capacity")
            ,col("cals_stdev_position_capacity")
            ,col("cals_median_stdev_position_capacity")
            ,col("position_capacity")
            ,col('position_item_vol')
            ,col('planogram_vol')
            ,col('plan_item_ratio')
            ,col("atc_median_dptcls")
            ,col("atc_mean_dptcls")
            ,col("atc_mean_locdptcls")
            ,col("atc_median_locdptcls")
            ,col("atc_stdev_locdptcls")
            ,col('coef_var')
            ,col('coef_var_sub')
            ,col("zscore_median")
            ,col("atc_zscore_dptcls_NULL")
            ,col("atc_zscore_locdptcls_NULL")
            ,col("atc_zscore_locdptcls_mch_NULL")
            ,col("atc_zscore_dptcls")
            ,col("atc_zscore_locdptcls")
            ,col("atc_zscore_locdptcls_mch")
            ,col("scaled_iso")
            ,col("scaled_svm")
        )
        self.spark.sql('drop table if exists stg_sc_atcad.oozie_iso_svm_results1')
        atc_anomaly_out.write.mode('overwrite').saveAsTable('stg_sc_atcad.oozie_iso_svm_results1')
        logger.info("ISO_SVM applied on data")

    #query_data gets the data from stg_sc_atcad.oozie_prep_results1
    def query_data(self):
        query = '''
            select a.*,
                concat(co_loc_ref_i, "_", dept_class) as loc_dept_cls,
                concat(co_loc_ref_i, "_", subclass_id) as loc_sub_clas,
                concat(co_loc_ref_i, "_", dept_class,"_", position_merch_style) as loc_deptcls_mch,
                cast(((fixture_depth*fixture_height*fixture_width)/1728) as decimal (10,3)) as fix_vol
            from stg_sc_atcad.oozie_prep_results1 as a
            where atc_value is NOT NULL'''
        self.query_info = self.spark.sql(query)
        logger.info("Query obtained")

    #startISOSVM starts the code for running the ISO-SVM algorithm
    def startISOSVM(self):
        self.query_data()
        self.iso_svm_data()

#Code to run to get data
pull = ISO_SVM()
pull.startISOSVM()
