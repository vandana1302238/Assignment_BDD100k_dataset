# Data analysis execution Steps

## Step 1:
Place the label files and give the respective train and val json files

## Step 2:
To execute docker file, build and run as per the following instructions:

```
docker build -t data_analysis .
 
docker run --rm -it data_analysis bash -v /home/labels:/home/labels
```

## Step 3:
To execute the code:
```
 
python bdd100k_data_analysis.py --data-dir /home/labels --output-results output_results
 ```