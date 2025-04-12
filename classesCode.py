import time
import logging
import threading
import numpy as np
import psutil
import warnings
import os
from joblib import Parallel, delayed
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')  

logger = logging.getLogger(__name__)

logger.propagate = False

# Hanya tambahkan handler jika belum ada
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

class SystemMonitor(threading.Thread):
    """Thread untuk memonitor penggunaan sumber daya sistem"""
    def __init__(self, interval=1):
        super().__init__()
        threading.Thread.__init__(self)
        self.interval = interval
        self.running = True
        self.cpu_usage = []
        self.memory_usage = []
        self.start_time = time.time()
        self.daemon	= True
        
    def run(self):
        while self.running:
            cpu = psutil.cpu_percent(interval=self.interval)
            memory = psutil.virtual_memory().percent
            self.cpu_usage.append(cpu)
            self.memory_usage.append(memory)
            
            logger.debug(f"Monitor: CPU {cpu}% | Memory {memory}%")
            
    def stop(self):
        self.running = False
        self.join()
        
    def get_stats(self):
        return {
            'avg_cpu': np.mean(self.cpu_usage),
            'max_cpu': np.max(self.cpu_usage),
            'avg_memory': np.mean(self.memory_usage),
            'max_memory': np.max(self.memory_usage),
            'duration': time.time() - self.start_time
        }
    def __getstate__(self):
        # Hentikan thread dan hapus atribut internal
        self.stop()
        state = self.__dict__.copy()
        for attr in ['_stderr', '_Thread__kwargs', '_Thread__args', '_tstate_lock']:
            if attr in state:
                del state[attr]
        return state

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, task='regression'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.task = task  # 'regression' or 'classification'
        self.tree = None
        self.feature_indices = None

    def calculate_impurity(self, y):
        """Menghitung impurity berdasarkan task"""
        if len(y) == 0:
            return 0
            
        y = np.asarray(y).flatten()
        if self.task == 'regression':
            # Mean Squared Error untuk regresi
            return np.mean((y - np.mean(y)) ** 2)
        else:
            # Gini Impurity untuk klasifikasi
            classes = np.unique(y)
            gini = 1.0
            for c in classes:
                p = np.sum(y == c) / len(y)
                gini -= p ** 2
            return gini

    def find_best_split(self, X, y):
        """Mencari split terbaik untuk dataset"""
        X = np.asarray(X)
        y = np.asarray(y).flatten()  # Pastikan y adalah 1D array
        
        n_samples, n_features = X.shape
        
        if n_samples < self.min_samples_split or n_features == 0:
            return None, None
            
        current_impurity = self.calculate_impurity(y)
        best_gain = 0.0
        best_feature, best_threshold = None, None
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            if len(unique_values) <= 1:
                continue
            if len(unique_values) > 10:
                percentiles = np.linspace(10, 90, num=10)
                thresholds = np.percentile(feature_values, percentiles)
            else:
                # Gunakan titik tengah antara nilai unik yang berurutan
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                # Buat mask boolean
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                if n_left < self.min_samples_split or n_right < self.min_samples_split:
                       continue
                left_impurity = self.calculate_impurity(y[left_mask])
                right_impurity = self.calculate_impurity(y[right_mask])
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples
            
                            # Hitung information gain
                gain = current_impurity - weighted_impurity
            
                            # Update jika gain lebih baik
                if gain > best_gain:
                       best_gain = gain
                       best_feature = feature_idx
                       best_threshold = threshold
        return best_feature, best_threshold

    def fit(self, X, y):
        """Fit decision tree to training data"""
        if len(X) == 0 or len(y) == 0 or len(X[0]) == 0:
            logger.warning("Empty training data!")
            self.tree = {'prediction': 0 if self.task == 'regression' else 0}
            return self
            
        logger.info(f"Building decision tree (max_depth={self.max_depth}, min_samples_split={self.min_samples_split})")
        self.tree = self._build_tree_recursive(X, y, depth=0)
        
        # Log statistik tree
        tree_depth = self.calculate_tree_depth()
        tree_size = self.calculate_tree_size()
        logger.info(f"Tree built: depth={tree_depth}, size={tree_size} nodes")
        
        return self
    
    def _build_tree_recursive(self, X, y, depth=0):
        """Build decision tree recursively"""
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = len(X)
        
        # Stopping criteria
        if (n_samples < self.min_samples_split or 
            depth >= self.max_depth or 
            len(set(y)) <= 1):  # Homogeneous target values
            
            # Create leaf node
            if self.task == 'regression':
                return {'prediction': np.mean(y)}
            else:  # classification
                classes, counts = np.unique(y, return_counts=True)
                return {'prediction': classes[np.argmax(counts)]}
        
        # Find best split
        feature_idx, threshold = self.find_best_split(X, y)
        
        # If no valid split found, create a leaf node
        if feature_idx is None:
            if self.task == 'regression':
                return {'prediction': np.mean(y)}
            else:
                classes, counts = np.unique(y, return_counts=True)
                return {'prediction': classes[np.argmax(counts)]}
        
        # Create split indices
        left_indices = [i for i in range(n_samples) if X[i][feature_idx] <= threshold]
        right_indices = [i for i in range(n_samples) if X[i][feature_idx] > threshold]
        
        # If split doesn't actually divide the dataset, create a leaf node
        if not left_indices or not right_indices:
            if self.task == 'regression':
                return {'prediction': np.mean(y)}
            else:
                classes, counts = np.unique(y, return_counts=True)
                return {'prediction': classes[np.argmax(counts)]}
        
        # Recursive call for children
        left_X = [X[i] for i in left_indices]
        left_y = [y[i] for i in left_indices]
        right_X = [X[i] for i in right_indices]
        right_y = [y[i] for i in right_indices]
        
        # Create decision node
        node = {
            'feature': feature_idx,
            'threshold': threshold,
            'left': self._build_tree_recursive(left_X, left_y, depth + 1),
            'right': self._build_tree_recursive(right_X, right_y, depth + 1)
        }
        
        return node
    
    def predict(self, X):
         return np.array([self._predict_single(x) for x in X])
        
    def _predict_single(self, x):
        """Predict for a single sample"""
        node = self.tree
        
        while 'prediction' not in node:
            if x[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
                
        return node['prediction']
        
    def calculate_tree_depth(self, tree=None):
        """Hitung kedalaman maksimum pohon"""
        if tree is None:
            tree = self.tree
        if tree is None or 'prediction' in tree:
            return 0
        return 1 + max(
            self.calculate_tree_depth(tree['left']),
            self.calculate_tree_depth(tree['right'])
        )
    
    def calculate_tree_size(self, tree=None):
        """Hitung jumlah node dalam pohon"""
        if tree is None:
            tree = self.tree
        if tree is None or 'prediction' in tree:
            return 1
        return 1 + self.calculate_tree_size(tree['left']) + self.calculate_tree_size(tree['right'])

def train_tree(X_np, y_np, n_features, subset_size, max_depth, min_samples_split, task='regression'):
    start_time = time.time()
    X_np = np.asarray(X_np, dtype=np.float32)
    y_np = np.asarray(y_np, dtype=np.float32 if task == 'regression' else int)
    indices = np.random.choice(len(X_np), len(X_np), replace=True)
    feature_indices = np.random.choice(n_features, subset_size, replace=False)
    
    tree = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split, task=task)
    tree.fit(X_np[indices][:, feature_indices], y_np[indices])  # Ganti build_tree dengan fit
    
    depth = tree.calculate_tree_depth()
    size = tree.calculate_tree_size()
    tree_time = time.time() - start_time
    
    return {
        'tree_object': tree,
        'feature_indices': feature_indices,
        'stats': {'depth': depth, 'size': size, 'time': tree_time}
    }

class EnhancedRandomForest:
    def __init__(
        self, 
        n_trees=50, 
        max_depth=10, 
        min_samples_split=5, 
        feature_subset_ratio=0.7,
        num_threads=min(4, os.cpu_count() // 2),
        monitoring_interval=1,
        task='regression'  # Tambahkan parameter task
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subset_ratio = feature_subset_ratio
        self.num_threads = num_threads
        self.monitoring_interval = monitoring_interval
        self.trees = []
        self.task = task  # 'regression' atau 'classification'
        self.system_monitor = None
        
    def fit(self, X, y):
        X_np = np.array(X, dtype=np.float32)
        y_np = np.array(y, dtype=np.float32 if self.task == 'regression' else object)
        n_features = X_np.shape[1]
        subset_size = int(n_features * self.feature_subset_ratio)
        
        # Mulai system monitor
        self.system_monitor = SystemMonitor(interval=self.monitoring_interval)
        self.system_monitor.start()
        
        logger.info(f"Memulai training dengan {self.n_trees} pohon...")
        start_time = time.time()
        last_log_time = start_time
        
        # Persiapan argumen untuk parallel processing
        tree_args = [(X_np, y_np, n_features, subset_size, 
                     self.max_depth, self.min_samples_split, self.task) 
                    for _ in range(self.n_trees)]
        
        # Training paralel dengan joblib
        try:
            results = Parallel(n_jobs=self.num_threads)(
                delayed(train_tree)(*args) for args in tree_args
            )
            self.trees = results
            
            # Hitung statistik setelah selesai
            total_time = time.time() - start_time
            tree_times = [t['stats']['time'] for t in self.trees]
            tree_depths = [t['stats']['depth'] for t in self.trees]
            tree_sizes = [t['stats']['size'] for t in self.trees]
            
            logger.info("\n=== Training Summary ===")
            logger.info(f"Training data: X shape: {len(X)}×{len(X[0])}, y range: {min(y)}-{max(y)}")
            logger.info(f"Total Training Time: {total_time:.2f} seconds")
            logger.info(f"Average Tree Time: {np.mean(tree_times):.4f} seconds")
            logger.info(f"Fastest Tree: {np.min(tree_times):.4f} seconds")
            logger.info(f"Slowest Tree: {np.max(tree_times):.4f} seconds")
            logger.info(f"Average Tree Depth: {np.mean(tree_depths):.1f}")
            logger.info(f"Average Tree Size: {np.mean(tree_sizes):.1f} nodes")
            logger.info(f"Training Speed: {self.n_trees/total_time:.2f} trees/second")
            
        except Exception as e:
            logger.error(f"Error selama training: {str(e)}")
            raise
        finally:
            # Stop monitor dan tampilkan laporan
            self.system_monitor.stop()
            sys_stats = self.system_monitor.get_stats()
            
            logger.info("\n=== System Usage Summary ===")
            logger.info(f"Average CPU Usage: {sys_stats['avg_cpu']:.1f}%")
            logger.info(f"Peak CPU Usage: {sys_stats['max_cpu']:.1f}%")
            logger.info(f"Average Memory Usage: {sys_stats['avg_memory']:.1f}%")
            logger.info(f"Peak Memory Usage: {sys_stats['max_memory']:.1f}%")
            logger.info(f"Total Duration: {sys_stats['duration']:.2f} seconds")
        
        return self

    def predict(self, X):
        X_np = np.array(X, dtype=np.float32)
        n_samples = len(X_np)
        all_predictions = np.zeros((n_samples, len(self.trees)))

        logger.info(f"Memulai prediksi {n_samples} sampel...")
        start_time = time.time()
        
        for i, tree_dict in enumerate(self.trees):
            tree = tree_dict['tree_object']
            feature_indices = tree_dict['feature_indices'] 
            
            # Log progress prediksi setiap 10 pohon
            if (i + 1) % 10 == 0 or (i + 1) == len(self.trees):
                elapsed = time.time() - start_time
                logger.info(f"Diproses {i + 1}/{len(self.trees)} pohon | Waktu: {elapsed:.2f}s")

            X_subset = X_np[:, feature_indices]
            all_predictions[:, i] = tree.predict(X_subset)

        logger.info("Prediksi selesai")
        
        if self.task == 'regression':
            return np.mean(all_predictions, axis=1)
        else:
            # Untuk klasifikasi, ambil modus (nilai yang paling sering muncul)
            return np.array([np.bincount(row.astype(int)).argmax() for row in all_predictions])

    def calculate_feature_importance(self):
        if not hasattr(self, 'features'):
            # Jika fitur tidak diset, ambil dari tree pertama
            n_features = len(self.trees[0]['feature_indices'])
            self.features = [f'feature_{i}' for i in range(n_features)]

        n_features = len(self.features)
        importances = np.zeros(n_features)
        split_counts = np.zeros(n_features)

        for tree_dict in self.trees:
            if not isinstance(tree_dict, dict):
                continue

            tree = tree_dict['tree_object']
            feature_indices = tree_dict['feature_indices']

            # Traverse tree dan hitung split
            stack = [tree.tree]
            while stack:
                node = stack.pop()

                if 'feature' in node:
                    global_idx = feature_indices[node['feature']]
                    if global_idx < n_features:
                        importances[global_idx] += 1  # Hitung frekuensi split
                        split_counts[global_idx] += 1

                if 'left' in node:
                    stack.append(node['left'])
                if 'right' in node:
                    stack.append(node['right'])

        # Normalisasi berdasarkan jumlah pohon
        if np.sum(split_counts) > 0:
            importances = importances / len(self.trees)
        else:
            # Fallback: distribusi merata jika tidak ada split
            importances = np.ones(n_features) / n_features

        return importances
    
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'system_monitor' in state:
            del state['system_monitor']
        return state
    def __setstate__(self, state):
								self.__dict__.update(state)
								self.system_monitor = None  # Inisialisasi ulang system_monitor saat unpickling

def normalize_data(X):
    X_np = np.array(X, dtype=np.float32)
    mean = np.mean(X_np, axis=0)
    std = np.std(X_np, axis=0)
    std[std == 0] = 1.0  # Hindari division by zero
    return (X_np - mean) / std

CATEGORY_DESCRIPTIONS = {
    0: "Tidak signifikan (Magnitudo < 2.5) - Biasanya tidak terasa",
    1: "Kerusakan ringan (Magnitudo 2.5-5.4) - Menyebabkan kerusakan ringan",
    2: "Kerusakan bangunan ringan (Magnitudo 5.5-6.0) - Mengakibatkan kerusakan ringan pada bangunan",
    3: "Banyak kerusakan (Magnitudo 6.1-6.9) - Menyebabkan banyak kerusakan di daerah padat penduduk",
    4: "Kerusakan serius (Magnitudo 7.0-7.9) - Gempa besar mengakibatkan kerusakan serius",
    5: "Menghancurkan (Magnitudo ≥8.0) - Dapat menghancurkan wilayah pusat gempa"
}

CATEGORY_NAMES = [
    "Tidak signifikan",
    "Kerusakan ringan",
    "Kerusakan bangunan ringan", 
    "Banyak kerusakan",
    "Kerusakan serius",
    "Menghancurkan"
]

class EarthquakePredictor:
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=5, feature_subset_ratio=0.7, num_threads=4):
        self.regressor = EnhancedRandomForest(
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            feature_subset_ratio=feature_subset_ratio,
            num_threads=num_threads,
            task='regression'
        )
        self.class_names = CATEGORY_NAMES
        self.category_descriptions = CATEGORY_DESCRIPTIONS
        self.features =	None
        self.norm_mean = None
        self.norm_std = None
        
    def fit(self, X, y):
        self.features = features
        X_np = np.array(X, dtype=np.float32)
        self.norm_mean = np.mean(X_np, axis=0)
        self.norm_std = np.std(X_np, axis=0)
        self.norm_std[self.norm_std == 0] = 1.0
        logger.info("Melatih model regresi...")
        self.regressor.fit(X, y)
        
    def predict(self, X):
        mag_pred = self.regressor.predict(X)
        class_pred = []
        for mag in mag_pred:
            if mag < 2.5:
                class_pred.append(0)
            elif 2.5 <= mag < 5.5:
                class_pred.append(1)
            elif 5.5 <= mag < 6.1:
                class_pred.append(2)
            elif 6.1 <= mag < 7.0:
                class_pred.append(3)
            elif 7.0 <= mag < 8.0:
                class_pred.append(4)
            else:
                class_pred.append(5)
        return mag_pred, np.array(class_pred)
    
    def predict_with_details(self, X):
        mag_pred = self.regressor.predict(X)
        results = []
        for mag in mag_pred:
            if mag < 2.5:
                category = 0
            elif 2.5 <= mag < 5.5:
                category = 1
            elif 5.5 <= mag < 6.1:
                category = 2
            elif 6.1 <= mag < 7.0:
                category = 3
            elif 7.0 <= mag < 8.0:
                category = 4
            else:
                category = 5
                
            results.append({
                'magnitude': mag,
                'category': category,
                'category_name': self.class_names[category],
                'description': self.category_descriptions[category]
            })
        return results
    def save_model(self, filename):
          """Save the trained model to disk"""
          with open(filename, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    @classmethod
    def load_model(cls, filename):
        """Load a trained model from disk"""
        with open(filename, 'rb') as f:
            return pickle.load(f)

