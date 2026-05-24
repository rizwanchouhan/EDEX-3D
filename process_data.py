import sys
from affectnet.AffectNetModule import AffectNetDataModule


def main(): 
    """
    Main function to process AffectNet dataset.

    Usage: python process_affectnet.py <input_folder> <output_folder> <optional_processed_subfolder> <optional_subset_index>
    input_folder: Folder where you downloaded and extracted AffectNet.
    output_folder: Folder where you want to process AffectNet.
    optional_processed_subfolder: If AffectNet is partly processed, specify the subfolder to finish processing.
    optional_subset_index: Index of subset of AffectNet if processing many parts in parallel (recommended).
    """

    # Get input and output folder paths from command line arguments
    downloaded_affectnet_folder = sys.argv[1]
    processed_output_folder = sys.argv[2]

    # Check if processed subfolder and subset index are provided
    if len(sys.argv) >= 3: 
        processed_subfolder = sys.argv[3]
    else: 
        processed_subfolder = None

    if len(sys.argv) >= 4: 
        sid = int(sys.argv[4])
    else: 
        sid = None

    # Initialize AffectNet data module
    dm = AffectNetDataModule(
            downloaded_affectnet_folder,
            processed_output_folder,
            processed_subfolder=processed_subfolder,
            mode="manual",
            scale=1.25,
            ignore_invalid=True,
            )

    # Detect landmarks and segment subset if subset index is provided
    if sid is not None:
        if sid >= dm.num_subsets: 
            print(f"Subset index {sid} is larger than the number of subsets. Terminating")
            return
        dm._detect_landmarks_and_segment_subset(dm.subset_size * sid, min((sid + 1) * dm.subset_size, len(dm.df)))
    else:
        # Otherwise, prepare data for processing
        dm.prepare_data() 

if __name__ == "__main__":
    main()