import os
import shutil

def clear_results():
    directories_to_clear = [
        "federated_evaluation_reports",
        "model_pickle",
        "scaler_data",
        "comparison_reports",
    ]
    
    files_to_remove = [
        "approach_comparison.csv",
        "centralized_model.pth",
        "centralized_scaler.pkl",
        "centralized_error_stats.pkl"
    ]

    print("🧹 Starting cleanup of previous experiment results...")

    for directory in directories_to_clear:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                print(f"✅ Deleted directory: {directory}")
            except Exception as e:
                print(f"❌ Failed to delete directory {directory}: {e}")
        else:
            print(f"ℹ️ Directory not found, skipping: {directory}")

    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"✅ Deleted file: {file}")
            except Exception as e:
                print(f"❌ Failed to delete file {file}: {e}")
        else:
            print(f"ℹ️ File not found, skipping: {file}")

    print("\n✨ Workspace cleared. You can now start fresh experiments.")

if __name__ == "__main__":
    confirm = input("Are you sure you want to clear ALL previous results and models? (y/n): ")
    if confirm.lower() == 'y':
        clear_results()
    else:
        print("Cleanup cancelled.")
