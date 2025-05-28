import pandas as pd
import numpy as np
import os
from feature_engineering import feature_engineering
import configuration

# Can be deleted just for testing, effectively legacy code for working out the programs flaws

def ensure_correct_directory():
    """Ensure we're in the correct directory for the script to work."""
    current_dir = os.getcwd()
    if not current_dir.endswith('Data_Processing'):
        # Try to find and change to Data_Processing directory
        if os.path.exists('Data_Processing'):
            os.chdir('Data_Processing')
            print(f"📁 Changed working directory to: {os.getcwd()}")
        else:
            print(f"⚠️  Current directory: {current_dir}")
            print("Please ensure you're running from the correct directory")
    else:
        print(f"📁 Working directory: {current_dir}")

def test_feature_engineering():
    """Test the feature engineering pipeline with actual data structure."""
    
    # Load the example data
    try:
        df = pd.read_csv('Testing_Feature_Data/Example.csv')
        print("✅ Successfully loaded Example.csv")
        print(f"Original data shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")
        
        # Show sample of original data
        print("\n📊 Sample of original data:")
        print(df.head(3))
        
        # Apply feature engineering
        print("\n🔧 Applying feature engineering pipeline...")
        engineered_df = feature_engineering(df)
        
        print(f"\n✅ Feature engineering completed!")
        print(f"Engineered data shape: {engineered_df.shape}")
        print(f"Number of new features created: {engineered_df.shape[1] - df.shape[1]}")
        
        # Show new features created
        new_features = [col for col in engineered_df.columns if col not in df.columns]
        print(f"\n🆕 New features created ({len(new_features)}):")
        for i, feature in enumerate(new_features, 1):
            print(f"{i:2d}. {feature}")
        
        # Show sample of key new features
        key_features = [
            'beaten_lengths_rank', 'field_size_category', 'nmfp_category', 
            'distance_category', 'going_category', 'layoff_category',
            'horse_win_rate', 'trainer_win_rate', 'jockey_win_rate',
            'speed_improvement', 'time_rank', 'combined_rating_rank'
        ]
        
        available_key_features = [f for f in key_features if f in engineered_df.columns]
        
        if available_key_features:
            print(f"\n🔍 Sample of key engineered features:")
            print(engineered_df[available_key_features].head())
        
        # Show some statistics
        print(f"\n📈 Feature Engineering Statistics:")
        print(f"- Original features: {df.shape[1]}")
        print(f"- New features: {len(new_features)}")
        print(f"- Total features: {engineered_df.shape[1]}")
        print(f"- Data completeness: {(1 - engineered_df.isnull().sum().sum() / (engineered_df.shape[0] * engineered_df.shape[1])) * 100:.1f}%")
        
        # Check for any remaining missing values
        missing_counts = engineered_df.isnull().sum()
        missing_features = missing_counts[missing_counts > 0]
        
        if len(missing_features) > 0:
            print(f"\n⚠️  Features with missing values:")
            for feature, count in missing_features.items():
                print(f"   {feature}: {count} missing ({count/len(engineered_df)*100:.1f}%)")
        else:
            print(f"\n✅ No missing values in engineered dataset!")
        
        # Save engineered data
        output_file = 'Testing_Feature_Data/engineered_example_data.csv'
        engineered_df.to_csv(output_file, index=False)
        print(f"\n💾 Engineered data saved to '{output_file}'")
        
        return engineered_df
        
    except FileNotFoundError:
        print("❌ Error: Could not find 'Testing_Feature_Data/Example.csv'")
        print("Please ensure the file exists in the correct location.")
        return None
    except Exception as e:
        print(f"❌ Error during feature engineering: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def analyze_feature_categories(df):
    """Analyze the different categories of features created."""
    if df is None:
        return
    
    #Load the configuration
    config = configuration.Config()
    
    print("\n🏷️  Feature Categories Analysis:")
    
    # Categorize features
    categories = {
        'Original': [],
        'Rankings': [],
        'Categories': [],
        'Rates/Percentages': [],
        'Improvements': [],
        'Temporal': [],
        'Market': [],
        'Performance': []
    }
    
    #Getting Original Columns
    original_cols = config.orignal_cols
    
    for col in df.columns:
        if col in original_cols:
            categories['Original'].append(col)
        elif 'rank' in col.lower():
            categories['Rankings'].append(col)
        elif 'category' in col.lower():
            categories['Categories'].append(col)
        elif 'rate' in col.lower() or 'pct' in col.lower() or 'percentage' in col.lower():
            categories['Rates/Percentages'].append(col)
        elif 'improvement' in col.lower() or 'improving' in col.lower():
            categories['Improvements'].append(col)
        elif any(word in col.lower() for word in ['race_', 'time_', 'season', 'layoff']):
            categories['Temporal'].append(col)
        elif any(word in col.lower() for word in ['odds', 'prob', 'sp']):
            categories['Market'].append(col)
        else:
            categories['Performance'].append(col)
    
    for category, features in categories.items():
        if features:
            print(f"\n{category} ({len(features)}):")
            for feature in features[:10]:  # Show first 10
                print(f"  • {feature}")
            if len(features) > 10:
                print(f"  ... and {len(features) - 10} more")

if __name__ == "__main__":
    print("🏇 Horse Racing Feature Engineering Test")
    print("=" * 50)
    
    # Ensure we're in the correct directory
    ensure_correct_directory()
    
    engineered_data = test_feature_engineering()
    analyze_feature_categories(engineered_data)
    
    print("\n" + "=" * 50)
    print("✅ Test completed!") 