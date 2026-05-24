import pandas as pd
import numpy as np
import os
import sys
import joblib

# Windows 终端控制台打印 utf-8 可能会报错，重新配置 stdout
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
except ImportError:
    print("❌ 错误：未安装 rdkit。请运行 `pip install rdkit` 安装。")
    exit(1)

try:
    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    
    from sklearn.metrics import (
        classification_report, roc_auc_score, accuracy_score,
        confusion_matrix, ConfusionMatrixDisplay, roc_curve,
        precision_score, recall_score, f1_score
    )
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端，避免弹窗
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("❌ 错误：未安装必要模型库。请运行 `pip install scikit-learn xgboost lightgbm catboost matplotlib seaborn` 安装。")
    exit(1)

# 配置 matplotlib 支持中文显示和高清输出
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

def smiles_to_ecfp4(smiles, radius=2, n_bits=1024):
    """
    将 SMILES 转换为 ECFP4 分子指纹
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def main():
    data_file = 'data/HDAC6_Activity_ML_Ready_Data.csv'

    if not os.path.exists(data_file):
        print(f"❌ 找不到数据文件 {data_file}，请先运行数据清洗脚本。")
        return

    print(f"📥 正在读取数据: {data_file}")
    df = pd.read_csv(data_file)

    if 'Smiles' not in df.columns or 'Label' not in df.columns:
        print("❌ 数据缺少 'Smiles' 或 'Label' 列，请检查数据集！")
        return

    # ========================================
    # 1. 提取 1024 位 ECFP4 指纹
    # ========================================
    print("🧬 正在提取分子的 ECFP4 指纹 (1024 bits)，这可能需要一些时间...")

    X_list = []
    valid_indices = []

    for idx, row in df.iterrows():
        smi = row['Smiles']
        fp = smiles_to_ecfp4(smi, n_bits=1024)
        if np.sum(fp) > 0:
            X_list.append(fp)
            valid_indices.append(idx)

    X = np.array(X_list)
    y = df.iloc[valid_indices]['Label'].values

    print(f"✅ 特征提取完成！成功提取了 {len(X)} 个分子的指纹。特征维度: {X.shape[1]}")

    # ========================================
    # 2. 数据集划分 (80% 训练集, 20% 测试集, 分层抽样)
    # ========================================
    print("\n🔀 正在划分训练集和测试集 (80/20, stratify)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   训练集: {X_train.shape[0]} 个样本 (Label=1: {np.sum(y_train)}, Label=0: {len(y_train) - np.sum(y_train)})")
    print(f"   测试集: {X_test.shape[0]} 个样本 (Label=1: {np.sum(y_test)}, Label=0: {len(y_test) - np.sum(y_test)})")

    # 保存测试集信息，供后续 SHAP 分析和阳性基准评估脚本使用（避免数据泄露）
    test_smiles = df.iloc[valid_indices].iloc[
        train_test_split(range(len(X)), test_size=0.2, random_state=42, stratify=y)[1]
    ]['Smiles'].values
    test_set_df = pd.DataFrame({'Smiles': test_smiles, 'Label': y_test})
    test_set_path = os.path.join("ML", "models", "test_set_activity.csv")
    test_set_df.to_csv(test_set_path, index=False)
    print(f"   💾 测试集数据已保存至: {test_set_path} (供 SHAP 和基准评估使用)")

    # 计算正负样本比例，用于 XGBoost 的类别不平衡处理
    neg_count = len(y_train) - np.sum(y_train)
    pos_count = np.sum(y_train)
    scale_ratio = neg_count / pos_count
    print(f"   正负样本比例: 1:{scale_ratio:.1f} → XGBoost scale_pos_weight = {scale_ratio:.1f}")

    # ========================================
    # 3. 定义 8 个模型及其参数网格
    # ========================================
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_dir = "ML/models"
    os.makedirs(model_dir, exist_ok=True)

    models_config = {
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
            "param_grid": {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20]
            }
        },
        "SVM": {
            "model": SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
            "param_grid": {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            }
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1, scale_pos_weight=scale_ratio),
            "param_grid": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6]
            }
        },
        "LightGBM": {
            "model": lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, class_weight='balanced'),
            "param_grid": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [31, 63]
            }
        },
        "CatBoost": {
            "model": cb.CatBoostClassifier(random_state=42, verbose=0, thread_count=-1, auto_class_weights='Balanced'),
            "param_grid": {
                'iterations': [100, 200],
                'learning_rate': [0.01, 0.1],
                'depth': [4, 6]
            }
        },
        "k-NN": {
            "model": KNeighborsClassifier(n_jobs=-1),
            "param_grid": {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        },
        "Naive Bayes": {
            "model": GaussianNB(),
            "param_grid": {}
        },
        "MLP": {
            "model": MLPClassifier(random_state=42, max_iter=500),
            "param_grid": {
                'hidden_layer_sizes': [(50,), (100,)],
                'alpha': [0.0001, 0.001]
            }
        }
    }

    # ========================================
    # 4. GridSearchCV 训练与评估
    # ========================================
    results = {}
    print("\n🚀 正在开始 8 个模型的 5折交叉验证调参及训练...")

    best_estimators = {}
    
    for name, config in models_config.items():
        print(f"\n[{name}] 进行网格搜索调参...")
        
        scoring_metrics = {'AUC': 'roc_auc', 'ACC': 'accuracy', 'F1': 'f1'}
        
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["param_grid"],
            cv=cv,
            scoring=scoring_metrics,
            refit='AUC', # 依然以 AUC 为标准来挑选最佳参数
            n_jobs=1, # 强制单进程避免内存爆炸
            verbose=0
        )
        try:
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            best_estimators[name] = best_model
            
            # --- Extract CV scores ---
            best_index = grid.best_index_
            
            cv_auc_scores = [grid.cv_results_[f'split{i}_test_AUC'][best_index] for i in range(grid.n_splits_)]
            cv_mean_auc = grid.cv_results_['mean_test_AUC'][best_index]
            cv_std_auc = grid.cv_results_['std_test_AUC'][best_index]
            
            cv_mean_acc = grid.cv_results_['mean_test_ACC'][best_index]
            cv_std_acc = grid.cv_results_['std_test_ACC'][best_index]
            
            cv_mean_f1 = grid.cv_results_['mean_test_F1'][best_index]
            cv_std_f1 = grid.cv_results_['std_test_F1'][best_index]
                
            # --- Predict on Train Set ---
            y_train_pred = best_model.predict(X_train)
            if hasattr(best_model, "predict_proba"):
                y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
            else:
                y_train_pred_proba = best_model.decision_function(X_train)
                
            train_acc = accuracy_score(y_train, y_train_pred)
            train_prec = precision_score(y_train, y_train_pred, zero_division=0)
            train_rec = recall_score(y_train, y_train_pred, zero_division=0)
            train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
            try:
                train_auc = roc_auc_score(y_train, y_train_pred_proba)
            except ValueError:
                train_auc = float('nan')

            # Predict on Test Set
            y_pred = best_model.predict(X_test)
            if hasattr(best_model, "predict_proba"):
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = best_model.decision_function(X_test)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                auc = float('nan')
                
            results[name] = {
                'Train_Accuracy': train_acc,
                'Train_Precision': train_prec,
                'Train_Recall': train_rec,
                'Train_F1': train_f1,
                'Train_AUC': train_auc,
                'Test_Accuracy': acc,
                'Test_Precision': prec,
                'Test_Recall': rec,
                'Test_F1': f1,
                'Test_AUC': auc,
                'CV_AUC_Scores': cv_auc_scores,
                'CV_Mean_AUC': cv_mean_auc,
                'CV_Std_AUC': cv_std_auc,
                'CV_Mean_ACC': cv_mean_acc,
                'CV_Std_ACC': cv_std_acc,
                'CV_Mean_F1': cv_mean_f1,
                'CV_Std_F1': cv_std_f1,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'model': best_model
            }
            print(f"✅ {name} 完成 - 最优参数: {grid.best_params_ if config['param_grid'] else 'N/A'}, 测试集 AUC: {auc:.4f}")
        except Exception as e:
            print(f"❌ {name} 训练出错: {e}")

    # ========================================
    # 5. 生成对比表格
    # ========================================
    print("\n📊 ============ 独立测试集评估对比表 ============")
    
    # 提取用于终端显示的指标
    table_data = []
    excel_data = []
    for name, res in results.items():
        table_data.append({
            'Model': name,
            'Accuracy': res['Test_Accuracy'],
            'Precision': res['Test_Precision'],
            'Recall': res['Test_Recall'],
            'F1-score': res['Test_F1'],
            'ROC-AUC': res['Test_AUC']
        })
        excel_data.append({
            'Model': name,
            'CV_Mean_AUC': res['CV_Mean_AUC'],
            'CV_Std_AUC': res['CV_Std_AUC'],
            'CV_Mean_ACC': res['CV_Mean_ACC'],
            'CV_Std_ACC': res['CV_Std_ACC'],
            'CV_Mean_F1': res['CV_Mean_F1'],
            'CV_Std_F1': res['CV_Std_F1'],
            'Train_AUC': res['Train_AUC'],
            'Train_Accuracy': res['Train_Accuracy'],
            'Train_F1': res['Train_F1'],
            'Train_Precision': res['Train_Precision'],
            'Train_Recall': res['Train_Recall'],
            'Test_AUC': res['Test_AUC'],
            'Test_Accuracy': res['Test_Accuracy'],
            'Test_F1': res['Test_F1'],
            'Test_Precision': res['Test_Precision'],
            'Test_Recall': res['Test_Recall']
        })

    metrics_df = pd.DataFrame(table_data)
    # 按 AUC 降序排列
    metrics_df = metrics_df.sort_values(by='ROC-AUC', ascending=False).reset_index(drop=True)
    
    excel_df = pd.DataFrame(excel_data)
    excel_df = excel_df.sort_values(by='Test_AUC', ascending=False).reset_index(drop=True)
    
    # 导出到 Excel/CSV
    excel_path = os.path.join(model_dir, "Model_Evaluation_Metrics_Comprehensive.xlsx")
    try:
        excel_df.to_excel(excel_path, index=False)
        print(f"✅ 详尽评估指标（CV、训练集、测试集）已导出至: {excel_path}")
    except Exception as e:
        csv_path = os.path.join(model_dir, "Model_Evaluation_Metrics_Comprehensive.csv")
        excel_df.to_csv(csv_path, index=False)
        print(f"⚠️ 无法导出 Excel (可能未安装 openpyxl)，已降级保存为 CSV: {csv_path}")

    # 绘制小提琴图
    print("\n🎻 正在绘制 5折交叉验证分数 (AUC) 的小提琴图...")
    try:
        plt.figure(figsize=(10, 6))
        
        cv_plot_data = []
        for name in metrics_df['Model']:
            for score in results[name]['CV_AUC_Scores']:
                cv_plot_data.append({'Model': name, 'CV_AUC': score})
                
        cv_df = pd.DataFrame(cv_plot_data)
        
        sns.violinplot(x='Model', y='CV_AUC', data=cv_df, inner='box', hue='Model', palette='Set3', legend=False)
        plt.title('8 大模型 5 折交叉验证 AUC 评分分布 (模型稳定性对比)', fontsize=14, pad=15)
        plt.xlabel('模型', fontsize=12)
        plt.ylabel('交叉验证 AUC 分数', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.5, axis='y')
        plt.tight_layout()
        
        violin_path = os.path.join(model_dir, "CV_5Fold_AUC_ViolinPlot.png")
        plt.savefig(violin_path)
        plt.close()
        print(f"✅ 小提琴图已保存至: {violin_path}")
    except Exception as e:
        print(f"❌ 绘制小提琴图出错: {e}")
    
    # 使用 pandas 格式化打印终端表格
    print("\n" + metrics_df.to_markdown(index=False, floatfmt=".4f"))
    print("================================================")

    # ========================================
    # 6. 选择最优模型并保存
    # ========================================
    best_name = metrics_df.iloc[0]['Model']
    best_auc = metrics_df.iloc[0]['ROC-AUC']
    best_model = results[best_name]['model']

    print(f"\n🏆 最优模型: {best_name} (ROC-AUC = {best_auc:.4f})")

    if best_auc > 0.85:
        print("✅ 最优模型 ROC-AUC > 0.85，达到任务指南要求！")
    else:
        print("⚠️ 最优模型 ROC-AUC 未达到 0.85 目标。")

    model_path = os.path.join(model_dir, "Best_HDAC6_Activity_Model.pkl")
    joblib.dump(best_model, model_path)
    print(f"💾 最优模型已保存至: {model_path}")

    # ========================================
    # 7. 可视化：ROC 曲线 (8 模型同图对比)
    # ========================================
    print("\n📈 正在绘制 8 模型 ROC 曲线对比图...")
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # 按 AUC 从高到低排序，使图例顺序与性能排名一致
    sorted_results = sorted(results.items(), key=lambda x: x[1]['Test_AUC'], reverse=True)
    
    for (name, res), color in zip(sorted_results, colors):
        fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {res['Test_AUC']:.4f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('HDAC6 Activity Prediction — 8 Models ROC Comparison', fontsize=14, pad=15)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    roc_path = os.path.join(model_dir, "ROC_All_Models_Activity_Comparison.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"✅ 合并 ROC 曲线图已保存至: {roc_path}")

    # ========================================
    # 8. 可视化：混淆矩阵 (分别保存)
    # ========================================
    print("📊 正在分别绘制 8 个模型的混淆矩阵图...")

    for name, res in results.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_test, res['y_pred'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Act(0)", "High Act(1)"])
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title(f'{name} — Confusion Matrix', fontsize=14, pad=15)
        plt.tight_layout()

        safe_name = name.lower().replace(" ", "_").replace("-", "")
        cm_path = os.path.join(model_dir, f"confusion_matrix_activity_{safe_name}.png")
        plt.savefig(cm_path)
        plt.close()
        
    print(f"✅ {len(results)} 个混淆矩阵图已分别保存至 {model_dir} 目录。")

    print("\n🎉 模型训练、评估与可视化全部完成！")


if __name__ == "__main__":
    main()
