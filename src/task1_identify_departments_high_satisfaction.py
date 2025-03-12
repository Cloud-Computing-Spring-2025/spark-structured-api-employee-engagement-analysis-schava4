# task1_identify_departments_high_satisfaction.py

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, round as spark_round

def initialize_spark(app_name="Task1_Identify_Departments"):
    """
    Initialize and return a SparkSession.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark

def load_data(spark, file_path):
    """
    Load the employee data from a CSV file into a Spark DataFrame.

    Parameters:
        spark (SparkSession): The SparkSession object.
        file_path (str): Path to the employee_data.csv file.

    Returns:
        DataFrame: Spark DataFrame containing employee data.
    """
    schema = "EmployeeID INT, Department STRING, JobTitle STRING, SatisfactionRating INT, EngagementLevel STRING, ReportsConcerns BOOLEAN, ProvidedSuggestions BOOLEAN"
    
    df = spark.read.csv(file_path, header=True, schema=schema)
    return df

def identify_departments_high_satisfaction(df):
    """
    Identify departments with more than 50% of employees having a Satisfaction Rating > 4 and Engagement Level 'High'.

    Parameters:
        df (DataFrame): Spark DataFrame containing employee data.

    Returns:
        DataFrame: DataFrame containing departments meeting the criteria with their respective percentages.
    """
    # Count total employees per department
    total_employees = df.groupBy("Department").agg(count("EmployeeID").alias("TotalEmployees"))
    
    # Count employees meeting the satisfaction criteria per department
    high_satisfaction = df.filter((col("SatisfactionRating") > 4) & (col("EngagementLevel") == "High"))
    high_satisfaction_count = high_satisfaction.groupBy("Department").agg(count("EmployeeID").alias("HighSatisfactionCount"))
    
    # Calculate the percentage of high satisfaction employees per department
    result_df = total_employees.join(high_satisfaction_count, "Department", "left").fillna(0)
    result_df = result_df.withColumn("HighSatisfactionPercentage", 
                                     spark_round((col("HighSatisfactionCount") / col("TotalEmployees")) * 100, 2))
    
    # Filter departments where more than 50% of employees meet the criteria
    result_df = result_df.filter(col("HighSatisfactionPercentage") > 5)
    
    return result_df.select("Department", "HighSatisfactionPercentage")

def write_output(result_df, output_path):
    """
    Write the result DataFrame to a CSV file, creating output directories if necessary.

    Parameters:
        result_df (DataFrame): Spark DataFrame containing the result.
        output_path (str): Path to save the output CSV file.

    Returns:
        None
    """
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    result_df.coalesce(1).write.csv(output_path, header=True, mode='overwrite')

def main():
    """
    Main function to execute Task 1.
    """
    # Initialize Spark
    spark = initialize_spark()
    
    # Define file paths
    input_file = "/workspaces/spark-structured-api-employee-engagement-analysis-schava4/input/employee_data.csv"
    output_file = "/workspaces/spark-structured-api-employee-engagement-analysis-schava4/outputs/task1/departments_high_satisfaction.csv"
    
    # Load data
    df = load_data(spark, input_file)
    
    # Perform Task 1
    result_df = identify_departments_high_satisfaction(df)
    
    # Write the result to CSV
    write_output(result_df, output_file)
    
    # Stop Spark Session
    spark.stop()

if __name__ == "__main__":
    main()
