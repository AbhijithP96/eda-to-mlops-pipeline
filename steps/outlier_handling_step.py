import pandas as pd
from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection, IQROutlierDetection
from zenml import step

@step
def outlier_detection_step(df: pd.DataFrame, column_name: str ,kwargs=None) -> pd.DataFrame:
    """Applies Outlier Detection and handling using the given methods"""

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    detection_method = kwargs['outlier_detection']

    if detection_method == 'zscore':
        zscore_thresh = kwargs['zscore_thresh'] if kwargs['zscore_thresh'] else 3.0
        detector = OutlierDetector(ZScoreOutlierDetection(zscore_thresh))

    elif detection_method == 'iqr':
        detector = OutlierDetector(IQROutlierDetection())

    else:
        raise ValueError(f'Unknown Outlier Detection Method: {detection_method}')
    
    df_numeric = df.select_dtypes(include=[int, float])
    handle_method = kwargs['handle_outliers']
    df_cleaned = detector.handle_outliers(df_numeric, method=handle_method)

    return df_cleaned