 
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import json
from sklearn.ensemble import RandomForestClassifier
import faiss
from sklearn.preprocessing import MinMaxScaler


def sample(df: pd.DataFrame, 
           category_column: str, 
           anomaly_ratio: float, 
           random_state: int = 42, 
           norm_rank: int=1,
           max_normal_sample=5000,
           max_anomaly_sample=1000) -> pd.DataFrame:
    """
    Sample a dataset into 'normal' and 'anomaly' subsets based on a categorical column.

    - The most frequent category is treated as the normal class.
    - Anomalies are randomly sampled from all other classes according to the specified ratio.

    Parameters:
    - df: Input DataFrame containing the data.
    - category_column: Name of the categorical column used to define classes.
    - anomaly_ratio: Fraction of normal samples to include as anomalies (e.g., 0.1 for 10%).
    - random_state: Optional seed for reproducibility.
    - norm_rank: choose the top-norm_rank frequent class as norm samples


    Returns:
    - A DataFrame consisting of all normal samples and a random subset of anomalies.
    """
    
    # Skip if the DataFrame has more than 100 columns
    if df.shape[1] > 100:
        return None
    
    # Remove non-numeric columns (excluding the category column) from the DataFrame
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in non_numeric_cols if col != category_column]
    df = df.drop(columns=non_numeric_cols)
    
    if df.shape[1] < 2:
        return None
    
    # Identify the normal class as the most frequent category
    counts = df[category_column].value_counts()
    # normal_class = counts.idxmax() # use the top 1
    norm_rank = min(len(counts), norm_rank)
    normal_class = counts.index[norm_rank - 1]

    # Split normals and potential anomalies
    normal_df = df[df[category_column] == normal_class].copy()
    n_norm = min(len(normal_df),max_normal_sample)
    normal_df = normal_df.sample(n=n_norm, random_state=random_state).copy()
    normal_df['anomaly_label'] = [0] * len(normal_df)
    other_df = df[df[category_column] != normal_class]

    # Determine number of anomalies to sample
    n_norm = len(normal_df)
    if anomaly_ratio >= 1.0:
        raise ValueError("anomaly_ratio must be less than 1")
    n_anom = int(anomaly_ratio * n_norm / (1.0 - anomaly_ratio))
    n_anom = min(n_anom, len(other_df), max_anomaly_sample)

    # Sample anomalies
    anomaly_df = other_df.sample(n=n_anom, random_state=random_state)
    anomaly_df['anomaly_label'] = [1] * len(anomaly_df)

    # Combine and shuffle
    combined = pd.concat([normal_df, anomaly_df]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    combined = combined.drop(category_column, axis=1, errors='ignore')
    return combined.astype(float).to_numpy()




classification_datasets = ['Amazon_employee_access', 'anneal', 'APSFailure', 'bank-marketing', 'Bank_Customer_Churn', 'Bioresponse', 'blood-transfusion-service-center', 'churn', 'coil2000_insurance_policies', 'credit-g', 'credit_card_clients_default', 'customer_satisfaction_in_airline', 'diabetes', 'E-CommereShippingData', 'Fitness_Club', 'GiveMeSomeCredit', 'hazelnut-spread-contaminant-detection', 'heloc', 'hiva_agnostic', 'HR_Analytics_Job_Change_of_Data_Scientists',
 'in_vehicle_coupon_recommendation', 'Is-this-a-good-customer',  'Marketing_Campaign', 'maternal_health_risk', 'MIC', 'NATICUSdroid', 'online_shoppers_intention', 'polish_companies_bankruptcy', 'qsar-biodeg', 'SDSS17', 'seismic-bumps', 'splice', 'students_dropout_and_academic_success', 'taiwanese_bankruptcy_prediction', 'website_phishing', 'jm1']



class ValidationDataset(Dataset):
    def __init__(
        self,
        max_feature_dim: int = 100,
        anomaly_ratio: float = 0.4,
    ):
        self.classification_datasets = classification_datasets
        self.max_feature_dim = max_feature_dim
        self.anomaly_ratio = anomaly_ratio

        # metadata: dataset_name -> target_feature
        metadata_path = ['/home/xding2/FoMo-0D-explore/data/TabArena/metadata/tabarena_dataset_metadata.csv',
                         '/data/haominwe/code/repurpose_real_data/data/TabLib_9_2/dataset_step2/metadata.csv',
                         #'/data/haominwe/code/repurpose_real_data/data/TabZilla/metadata/tabzilla_dataset_metadata.csv',
                         #'/data/haominwe/code/repurpose_real_data/data/OpenML-CC18/metadata/openml-cc18_dataset_metadata.csv',
                         #'/data/haominwe/code/repurpose_real_data/data/AutoML/metadata/openml-cc18_dataset_metadata.csv'
                         ]
        data_files = ['/home/xding2/FoMo-0D-explore/data/TabArena/datasets/',
                      '/data/haominwe/code/repurpose_real_data//data/TabLib_9_2/dataset_step2/data/',
                      #'/data/haominwe/code/repurpose_real_data/data/TabZilla/datasets/',
                      #'/data/haominwe/code/repurpose_real_data/data/OpenML-CC18/datasets/',
                      #'/data/haominwe/code/repurpose_real_data/data/AutoML/datasets/'
                      ]
        
        # dataset classes counts
        self.dataset_dict = {}
        self.cache = {}
        self.name_to_tgt = {}
        self.dataset_list = []  # List to store dataset names for indexing

        # Save dataset_dict, name_to_tgt, and dataset_list to files
        self.dataset_dict_file = 'val_dataset_dict.json'
        self.name_to_tgt_file = 'val_name_to_tgt.json'
        self.dataset_list_file = 'val_dataset_list.txt'

        def save_to_files():
            with open(self.dataset_dict_file, 'w') as f:
                json.dump(self.dataset_dict, f)
            with open(self.name_to_tgt_file, 'w') as f:
                json.dump(self.name_to_tgt, f)
            with open(self.dataset_list_file, 'w') as f:
                f.write('\n'.join(self.dataset_list))

        def load_from_files():
            if os.path.exists(self.dataset_dict_file):
                with open(self.dataset_dict_file, 'r') as f:
                    self.dataset_dict = json.load(f)
            if os.path.exists(self.name_to_tgt_file):
                with open(self.name_to_tgt_file, 'r') as f:
                    self.name_to_tgt = json.load(f)
            if os.path.exists(self.dataset_list_file):
                with open(self.dataset_list_file, 'r') as f:
                    self.dataset_list = f.read().splitlines()
        
        self.save_to_files = save_to_files
        self.num_episodes = 0
        
        # load_from_files()
        # self.num_episodes = len(self.dataset_list)
        # return 
        
        files = os.listdir('/home/xding2/FoMo-0D-explore/data/val_data_v2')
        
        for idx, meta_p in tqdm(enumerate(metadata_path)):
            meta_df = pd.read_csv(meta_p)
            data_root = data_files[idx]
            name_to_tgt_feature_dict = dict(zip(meta_df['dataset_name'], meta_df['target_feature']))
            if idx == 0:
                #first one, use all classification datasets
                for i in classification_datasets:
                    data_name = i
                    data_path = data_root + f'{data_name}.csv'
                    df = pd.read_csv(data_path)
                    
                    tgt_fea = name_to_tgt_feature_dict[data_name]
                    counts = df[tgt_fea].value_counts()
                    self.dataset_dict[data_path] = len(counts)
                    self.name_to_tgt[data_path] = tgt_fea
                                
                    
                    for j in range(len(counts)):
                        save_name = f"{data_path}%{j}"
                        save_name = save_name.split('/')[-1] + ".npz"
                        if save_name in files:
                            self.dataset_list.append(f"{data_path}%{j}")
                            self.num_episodes += 1
                            continue
                    
                        batch = sample(df, category_column=tgt_fea, anomaly_ratio=self.anomaly_ratio, random_state=42, norm_rank=j)
                        print('sampling done for dataset:', data_name, 'norm_rank:', j)
                        if batch is None:
                            continue
                        else:
                            X = torch.from_numpy(batch[:, :-1])
                            y = torch.from_numpy(batch[:, -1]).long()
                            if X.shape[0] > 1000 and (X.shape[1]>=1) and (not np.isinf(X).any()) and (not np.isnan(X).any()):
                                train_inliers, test_x, test_y = self.split_train_test(X, y)
                                train_inliers_raw = train_inliers
                                test_x_raw = test_x
                                
                                # Min-max normalize based on train_inliers statistics
                                scaler = MinMaxScaler()
                                train_inliers_normalized = scaler.fit_transform(train_inliers.numpy())
                                test_x_normalized = scaler.transform(test_x.numpy())
                                
                                # Convert back to tensors
                                train_inliers = torch.from_numpy(train_inliers_normalized).float()
                                test_x = torch.from_numpy(test_x_normalized).float()
                                
                                # Fit a KNN model using FAISS
                                d = train_inliers.shape[1]  # Dimensionality of the data
                                index = faiss.IndexFlatL2(d)  # L2 distance
                                index.add(train_inliers.float().numpy())  # Add training inliers to the index

                                # Predict distances for test data
                                distances, _ = index.search(test_x.float().numpy(), k=5)  # k=1 nearest neighbor
                                scores = np.mean(distances,axis=1).flatten()  # Use negative distance as the score (higher is more likely outlier)
                                
                                # Calculate AUC-ROC
                                auc_roc = roc_auc_score(test_y.numpy(), scores)  
                                print(f"Dataset: {data_name}, norm_rank: {j}, KNN AUC-ROC: {auc_roc:.4f}")
                                
                                
                                # Fit Isolation Forest only on training inliers
                                # clf = IsolationForest(random_state=42)
                                # clf.fit(train_inliers.numpy())  # Train only on normal samples
                                # scores = -clf.decision_function(test_x.numpy())  # Evaluate on test data
                                # auc_roc = roc_auc_score(test_y.numpy(), scores)
                                # print(f"Dataset: {data_name}, norm_rank: {j}, Trained IsoForest AUC-ROC: {auc_roc:.4f}")
                                
                                if auc_roc < 0.7 or auc_roc > 0.95:
                                    continue
                                self.dataset_list.append(f"{data_path}%{j}")
                                save_name = f"{data_path}%{j}"
                                save_name = save_name.split('/')[-1]
                                np.savez(f"/home/xding2/FoMo-0D-explore/data/val_data_v2/{save_name}.npz", train_x=train_inliers_raw.numpy(),test_x =test_x_raw.numpy(), y=test_y.numpy())
                                self.num_episodes += 1
            else:
                for i in list(name_to_tgt_feature_dict.keys())[1533:]:
                    data_name = i
                    if not data_name.endswith('.csv'):
                        data_path = data_root + f'{data_name}.csv'
                    else:
                        data_path = data_root + f'{data_name}'
                    df = pd.read_csv(data_path)

                    tgt_fea = name_to_tgt_feature_dict[data_name]
                    counts = df[tgt_fea].value_counts()
                    self.dataset_dict[data_path] = len(counts)
                    self.name_to_tgt[data_path] = tgt_fea
                    
                    for j in range(len(counts)):
                        save_name = f"{data_path}%{j}"
                        save_name = save_name.split('/')[-1] + ".npz"
                        if save_name in files:
                            self.dataset_list.append(f"{data_path}%{j}")
                            self.num_episodes += 1
                            continue
                        
                        batch = sample(df, category_column=tgt_fea, anomaly_ratio=self.anomaly_ratio,
                            random_state=42, norm_rank=j)
                        if batch is None:
                            continue
                        # Split features/labels
                        X = torch.from_numpy(batch[:, :-1])
                        y = torch.from_numpy(batch[:, -1]).long()
                        print('sampling done for dataset:', data_name, 'norm_rank:', j)
                        
                        if X.shape[0] > 1000 and (X.shape[1]>=1) and (not np.isinf(X).any()) and (not np.isnan(X).any()):
                            train_inliers, test_x, test_y = self.split_train_test(X, y)
                            print(train_inliers.shape, test_x.shape, test_y.shape)
                            train_inliers_raw = train_inliers
                            test_x_raw = test_x
                                
                            # Min-max normalize based on train_inliers statistics
                            scaler = MinMaxScaler()
                            train_inliers_normalized = scaler.fit_transform(train_inliers.numpy())
                            test_x_normalized = scaler.transform(test_x.numpy())
                            
                            # Convert back to tensors
                            train_inliers = torch.from_numpy(train_inliers_normalized).float()
                            test_x = torch.from_numpy(test_x_normalized).float()         
                                
                            # Fit a KNN model using FAISS
                            d = train_inliers.shape[1]  # Dimensionality of the data
                            index = faiss.IndexFlatL2(d)  # L2 distance
                            index.add(train_inliers.float().numpy())  # Add training inliers to the index

                            # Predict distances for test data
                            distances, _ = index.search(test_x.float().numpy(), k=5)  # k=1 nearest neighbor
                            scores = np.mean(distances,axis=1).flatten()  # Use negative distance as the score (higher is more likely outlier)

                            # Calculate AUC-ROC
                            auc_roc = roc_auc_score(test_y.numpy(), scores)  
                            print(f"Dataset: {data_name}, norm_rank: {j}, KNN AUC-ROC: {auc_roc:.4f}")
                            
                            # clf = IsolationForest(random_state=42)
                            # clf.fit(X)
                            # scores = -clf.decision_function(X)
                            # auc_roc = roc_auc_score(y.numpy(), scores)
                            # print(f"Dataset: {data_name}, norm_rank: {j}, IsoForest AUC-ROC: {auc_roc:.4f}")

                            # Fit Isolation Forest only on training inliers
                            # clf = IsolationForest(random_state=42)
                            # clf.fit(train_inliers.numpy())  # Train only on normal samples
                            # scores = -clf.decision_function(test_x.numpy())  # Evaluate on test data
                            # auc_roc = roc_auc_score(test_y.numpy(), scores)
                            # print(f"Dataset: {data_name}, norm_rank: {j}, Trained IsoForest AUC-ROC: {auc_roc:.4f}")
                            
                            if auc_roc < 0.7 or auc_roc > 0.95:
                                continue
                            self.dataset_list.append(f"{data_path}%{j}")
                            save_name = f"{data_path}%{j}"
                            save_name = save_name.split('/')[-1]
                            np.savez(f"/home/xding2/FoMo-0D-explore/data/val_data_v2/{save_name}.npz", train_x=train_inliers_raw.numpy(),test_x =test_x_raw.numpy(), y=test_y.numpy())
                            self.num_episodes += 1
        self.save_to_files()
    
        
    def split_train_test(self,X,y):
        inliers = X[y==0]
        outliers = X[y==1]
        seq_len = inliers.shape[0]
        single_eval_pos = int(seq_len * 0.5)
        
        num_inliers = single_eval_pos
        num_test_x = seq_len - single_eval_pos
        
        train_inliers= inliers[:single_eval_pos]
        test_inliers = inliers[single_eval_pos:]
        test_la = outliers
        test_x = torch.cat([test_inliers, test_la], dim=0)
        test_y = torch.tensor([0] * num_test_x + [1] * test_la.shape[0])
        return train_inliers, test_x, test_y
        
    
    def set_rank(self, rank):
        self.rank = rank
        
    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        if idx in self.cache: return self.cache[idx]
        # Allow referencing by index
        #print(self.dataset_list)
        dataset_name_rank = self.dataset_list[idx]
        dataset_name = dataset_name_rank.split('%')[0]
        rank = int(dataset_name_rank.split('%')[1])
            
        df = pd.read_csv(dataset_name)
        tgt = self.name_to_tgt[dataset_name]

        # Your sample() should return a torch.Tensor with label in the last column
        batch = sample(df, category_column=tgt, anomaly_ratio=self.anomaly_ratio,
                       random_state=42, norm_rank=rank)
        # Split features/labels
        X = torch.from_numpy(batch[:, :-1])
        y = torch.from_numpy(batch[:, -1]).long()
        self.cache[idx] = {"X": X, "y": y}

        # Transform + pad/truncate features to max_feature_dim
        #X = pfn_transform(eval_xs=X, max_feature_dim=self.max_feature_dim)  # -> (N, max_feature_dim)
        np.savez(f"/home/xding2/FoMo-0D-explore/data/val_data/{idx}.npz", X=X.numpy(), y=y.numpy())
        return {"X": X, "y": y}  # X: (N, D), y: (N,)




tabarena_classification_datasets = ['Amazon_employee_access', 'anneal', 'APSFailure', 'bank-marketing', 'Bank_Customer_Churn', 'Bioresponse', 'blood-transfusion-service-center', 'churn', 'coil2000_insurance_policies', 'credit-g', 'credit_card_clients_default', 'customer_satisfaction_in_airline', 'diabetes', 'Diabetes130US', 'E-CommereShippingData', 'Fitness_Club', 'GiveMeSomeCredit', 'hazelnut-spread-contaminant-detection', 'heloc', 'hiva_agnostic', 'HR_Analytics_Job_Change_of_Data_Scientists',
 'in_vehicle_coupon_recommendation', 'Is-this-a-good-customer', 'kddcup09_appetency', 'Marketing_Campaign', 'maternal_health_risk', 'MIC', 'NATICUSdroid', 'online_shoppers_intention', 'polish_companies_bankruptcy', 'qsar-biodeg', 'SDSS17', 'seismic-bumps', 'splice', 'students_dropout_and_academic_success', 'taiwanese_bankruptcy_prediction', 'website_phishing', 'jm1']

# Example usage:
if __name__ == "__main__":
    
    
    if 1: 
        dataset = ValidationDataset(max_feature_dim=100, anomaly_ratio=0.2)
        for i in dataset:
            print(i['X'].shape, i['y'].shape)
        print(f"Total valid episodes: {len(dataset)}")
    