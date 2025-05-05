## get aws account information
aws_region = "REGION"
sts_client = boto3.client("sts")
try:
    response = sts_client.get_caller_identity()
    aws_account_id = response["Account"]  
  except:
    raise Exception("Cannot determine aws account")

s3_client = boto3.client("s3", "REGION")

## Redshift interaction set up and module call 
redshift_data_client = RedshiftDataInteraction()
MEDIA_ROLE_ARN = f"arn:aws:iam::{aws_account_id}:role/LOCATION-redshift"
ROLE_ARN = redshift_data_client._role_arn

## Dynamic bucket naming 
temp_bucket = "temp-account-region"
support_bucket = "support-account-region"
landing_bucket = "landing-internal-account-region"

today_date = datetime.strftime(datetime.now(), "%Y-%m-%d")
log_output_folder = f'PATH/model_logs/dt={today_date}'
result_output_location = f's3://{landing_bucket}/PATH/TITLE/dt={today_date}'

results_file_name = 'TITLE.csv'





class RedshiftDataInteraction:
    __is_unit_test = False   ## since moto doesn't have full support for redshift-data, have to include conditional test
    def __init__(self, region='REGION'):
        self._config = self._read_secret_config()
        self._cluster_identifier = self._config['cluster_identifier']
        self._role_arn = self._config['role_arn']
        self._db_user = self._config['db_user']
        self._database = self._config['database']
        self._credentials = boto3.client("sts", region_name=region).assume_role(
            RoleArn=self._role_arn,
            RoleSessionName="dpx_client",
        )['Credentials']
        self._rsd_client = boto3.client(
            'redshift-data',
            region_name=region,
            aws_access_key_id=self._credentials['AccessKeyId'],
            aws_secret_access_key=self._credentials['SecretAccessKey'],
            aws_session_token=self._credentials['SessionToken']
        )

    def _read_secret_config(self):
        redshift_config = SecretManager().get_secret_value('LOCATION_redshift_config')
        return json.loads(redshift_config)

    def _poll_statement(self, stmt_id):
        while True:
            stmt_status = self._rsd_client.describe_statement(Id=stmt_id)
            logger.debug(f'Polling {stmt_id} status: {stmt_status["Status"]}')
            if stmt_status['Status'] in ['SUBMITTED', 'PICKED', 'STARTED'] and not self.__is_unit_test:
                time.sleep(5)
            elif stmt_status['Status'] == 'FINISHED':
                return True, stmt_status
            elif stmt_status['Status'] in ['ABORTED', 'FAILED']:
                raise Exception(stmt_status['Error'])
            if self.__is_unit_test:
                return True, {'HasResultSet': True, 'ResultRows': 3}

    def execute_query(self, query, page_size=100):
        response = self._rsd_client.execute_statement(
            ClusterIdentifier=self._cluster_identifier,
            DbUser=self._db_user,
            Database=self._database,
            Sql=query
        )
        stmt_id = response['Id']
        success, status = self._poll_statement(stmt_id)
        logger.debug(f'statement id: {stmt_id} - {success} - {status}')
        if success and status['HasResultSet'] and status.get('ResultRows', 0) > 0:
            yield from self._paginate_response(stmt_id, page_size)
        else:
            yield []

    def unload(self, query:str, unload_location:str, addtional_roles:list=None):
        query_role = self._role_arn
        if addtional_roles is not None and len(addtional_roles) > 0:
            query_role = ','.join([query_role, *addtional_roles])

        unload_query = f"""
                unload ('{query}') to '{unload_location}'
                iam_role '{query_role}'
                FORMAT AS PARQUET
                MAXFILESIZE 100 MB
                ALLOWOVERWRITE;
            """
        print(f"Executing query: {unload_query}")
        response = self.execute_query(unload_query)
        return list(response)

    def _paginate_response(self, stmt_id, page_size):
        response = self._rsd_client.get_statement_result(Id=stmt_id)
        columns = [col['name'] for col in response['ColumnMetadata']]
        if self.__is_unit_test:
            response['TotalNumRows'] = len(response['Records'])  ## mock response doesn't return this value
        total_rows = response['TotalNumRows']
        current_row = 0
        while current_row < total_rows:
            next_token = response.get('NextToken')
            rows = []
            for record in response['Records']:
                current_row += 1
                values = []
                for col in record:
                    d_type, value = next(iter(col.items()))
                    values.append(None if d_type == 'isNull' and value else value)
                rows.append(dict(zip(columns, values)))
                if len(rows) == page_size:
                    yield rows
                    rows = []
            if next_token:
                print(f'Paginating next token {next_token}')
                response = self._rsd_client.get_statement_result(Id=stmt_id, NextToken=next_token)
        yield rows



          
query = '''
SELECT * FROM SCHEMA.TABLE
'''


          
###unload data from s3
unload_path = f"{result_output_location}/"

redshift_data_client.unload(query=query, unload_location=unload_path, addtional_roles=[MEDIA_ROLE_ARN])

data = spark.read.parquet(f"s3://{landing_bucket}/PATH/TITLE/dt={today_date}")



### safely toPandas()
data1 = data.toPandas()
df = data1.copy()
