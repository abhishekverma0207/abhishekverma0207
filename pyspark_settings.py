spark.conf.set('spark.executor.memory', '15g')
spark.conf.set('spark.driver.memory', '40g')
spark.conf.set('spark.executor.instances', '100')
spark.conf.set('spark.executor.cores', '10')
spark.conf.set('spark.sql.shuffle.partitions', '300')
spark.conf.set('spark.default.parallelism', '300')
spark.conf.set('spark.shuffle.io.maxRetries','5')

# spark.conf.set('spark.sql.autoBroadcastJoinThreshold', '-1')
spark.conf.set('spark.python.worker.memory','5G')
spark.conf.set('spark.sql.execution.arrow.enabled', 'true')
# spark.conf.set('spark.memory.fraction','0.7')

spark.conf.set('spark.shuffle.service.enabled', 'false')
spark.conf.set('spark.dynamicAllocation.enabled', 'false')
# spark.conf.set('spark.dynamicAllocation.minExecutors','1')
# spark.conf.set('spark.dynamicAllocation.maxExecutors', '100')

spark.conf.set('yarn.nodemanager.vmem-check-enabled','false')
spark.conf.set('spark.maximizeResourceAllocation','true')

spark.conf.set('spark.executor.extraJavaOptions', '-Xmx24g')
spark.conf.set('yarn.scheduler.minimum-allocation-mb','10000')

spark.conf.set('spark.yarn.executor.memoryOverhead', '38G')
spark.conf.set('spark.executor.memoryOverhead', '20g')
spark.conf.set('spark.driver.memoryOverhead', '10g')

spark.conf.set('spark.submit.deployMode', 'client')


spark.conf.set('spark.memory.offHeap.enabled', 'true')
spark.conf.set('spark.memory.offHeap.size', '6g')
