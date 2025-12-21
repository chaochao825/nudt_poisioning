import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from data import get_data
from model import get_model
import json

def sse_print(event: str, data: dict) -> str:
    """
    SSE 打印
    :param event: 事件名称
    :param data: 事件数据（字典或能被 json 序列化的对象）
    :return: SSE 格式字符串
    """
    # 将数据转成 JSON 字符串
    json_str = json.dumps(data, ensure_ascii=False, default=lambda obj: obj.item() if isinstance(obj, np.generic) else obj)
    
    # 按 SSE 协议格式拼接
    message = f"event: {event}\n" \
              f"data: {json_str}\n\n"
    print(message, flush=True, end='')
    return message

class AnomalyDetector:
    """
    使用Isolation Forest进行异常检测的类
    主要用于检测数据投毒攻击
    """
    
    def __init__(self, contamination=0.1):
        """
        初始化异常检测器
        
        Args:
            contamination: 预期的异常值比例，默认为0.1 (10%)
        """
        self.contamination = contamination
        self.iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
    def extract_features(self, x_data):
        """
        从图像数据中提取特征用于异常检测
        使用简单的统计特征：均值、标准差、最大值、最小值等
        
        Args:
            x_data: 输入数据，形状为(N, C, H, W)
            
        Returns:
            提取的特征，形状为(N, num_features)
        """
        # 确保数据在正确的设备上并转换为numpy数组
        if isinstance(x_data, torch.Tensor):
            x_data = x_data.cpu().numpy()
        
        # 如果是图像数据，展平通道维度
        if len(x_data.shape) == 4:
            n_samples = x_data.shape[0]
            # 计算每张图像的基本统计特征
            means = np.mean(x_data, axis=(1, 2, 3))
            stds = np.std(x_data, axis=(1, 2, 3))
            mins = np.min(x_data, axis=(1, 2, 3))
            maxs = np.max(x_data, axis=(1, 2, 3))
            
            # 计算每个通道的统计特征
            channel_means = np.mean(x_data, axis=(2, 3))  # (N, C)
            channel_stds = np.std(x_data, axis=(2, 3))    # (N, C)
            
            # 合并所有特征
            features = np.column_stack([
                means, stds, mins, maxs,
                channel_means,
                channel_stds
            ])
        else:
            # 如果已经是展平的数据
            features = x_data
            
        return features
    
    def fit(self, x_train):
        """
        训练Isolation Forest模型
        
        Args:
            x_train: 训练数据
        """
        # 提取特征
        features = self.extract_features(x_train)
        
        # 训练模型
        self.iso_forest.fit(features)
        sse_print("training_complete", {
            "message": f"Isolation Forest模型训练完成，预期异常比例: {self.contamination*100:.1f}%",
            "contamination": self.contamination
        })
        
    def predict(self, x_data):
        """
        预测数据点是否为异常值
        
        Args:
            x_data: 待检测的数据
            
        Returns:
            预测结果: 1表示正常，-1表示异常
        """
        # 提取特征
        features = self.extract_features(x_data)
        
        # 预测
        predictions = self.iso_forest.predict(features)
        return predictions
    
    def decision_function(self, x_data):
        """
        计算异常分数（越小越异常）
        
        Args:
            x_data: 待检测的数据
            
        Returns:
            异常分数
        """
        # 提取特征
        features = self.extract_features(x_data)
        
        # 计算异常分数
        scores = self.iso_forest.decision_function(features)
        return scores
    
    def evaluate_poisoning(self, x_clean, x_suspected_poisoned):
        """
        评估疑似中毒数据中的异常比例
        
        Args:
            x_clean: 干净的基准数据
            x_suspected_poisoned: 可疑的可能中毒数据
            
        Returns:
            检测结果统计
        """
        # 提取特征
        clean_features = self.extract_features(x_clean)
        suspected_features = self.extract_features(x_suspected_poisoned)
        
        # 在干净数据上预测
        clean_predictions = self.iso_forest.predict(clean_features)
        clean_anomalies = np.sum(clean_predictions == -1)
        clean_anomaly_rate = clean_anomalies / len(clean_predictions)
        
        # 在可疑数据上预测
        suspected_predictions = self.iso_forest.predict(suspected_features)
        suspected_anomalies = np.sum(suspected_predictions == -1)
        suspected_anomaly_rate = suspected_anomalies / len(suspected_predictions)
        
        # 输出详细指标
        metrics_data = self._prepare_metrics_data(
            clean_anomaly_rate, clean_anomalies, len(clean_predictions),
            suspected_anomaly_rate, suspected_anomalies, len(suspected_predictions)
        )
        
        sse_print("evaluation_metrics", metrics_data)
        
        detection_result = {
            'clean_anomaly_rate': clean_anomaly_rate,
            'suspected_anomaly_rate': suspected_anomaly_rate,
            'clean_anomalies': clean_anomalies,
            'suspected_anomalies': suspected_anomalies,
            'anomaly_increase': suspected_anomaly_rate - clean_anomaly_rate
        }
        
        if suspected_anomaly_rate > clean_anomaly_rate:
            warning_msg = "警告: 可疑数据中的异常比例显著高于干净数据，可能存在投毒攻击!"
            increase_msg = f"异常比例增加了 {(suspected_anomaly_rate/clean_anomaly_rate - 1)*100:.1f}%"
            sse_print("poisoning_detected", {
                "warning": warning_msg,
                "increase_info": increase_msg,
                "is_attack_detected": True
            })
        else:
            info_msg = "可疑数据中的异常比例在正常范围内，未发现明显投毒迹象。"
            sse_print("no_poisoning_detected", {
                "info": info_msg,
                "is_attack_detected": False
            })
            
        return detection_result
    
    def _prepare_metrics_data(self, clean_rate, clean_anomalies, clean_total,
                             suspect_rate, suspect_anomalies, suspect_total):
        """
        准备指标数据用于SSE输出
        """
        return {
            "clean_data": {
                "total": clean_total,
                "anomalies": clean_anomalies,
                "anomaly_rate": float(clean_rate),
                "anomaly_percentage": float(clean_rate * 100)
            },
            "suspected_data": {
                "total": suspect_total,
                "anomalies": suspect_anomalies,
                "anomaly_rate": float(suspect_rate),
                "anomaly_percentage": float(suspect_rate * 100)
            },
            "comparison": {
                "rate_difference": float(suspect_rate - clean_rate),
                "percentage_difference": float((suspect_rate - clean_rate) * 100)
            }
        }
        
    def get_detailed_stats(self, x_data):
        """
        获取详细的统计数据
        
        Args:
            x_data: 数据
            
        Returns:
            详细统计数据
        """
        # 提取特征和计算异常分数
        features = self.extract_features(x_data)
        scores = self.iso_forest.decision_function(features)
        predictions = self.iso_forest.predict(features)
        
        normal_count = np.sum(predictions == 1)
        anomaly_count = np.sum(predictions == -1)
        
        normal_scores = scores[predictions == 1]
        anomaly_scores = scores[predictions == -1]
        
        stats = {
            'total_samples': len(predictions),
            'normal_samples': int(normal_count),
            'anomaly_samples': int(anomaly_count),
            'normal_score_mean': float(np.mean(normal_scores)) if len(normal_scores) > 0 else 0,
            'normal_score_std': float(np.std(normal_scores)) if len(normal_scores) > 0 else 0,
            'anomaly_score_mean': float(np.mean(anomaly_scores)) if len(anomaly_scores) > 0 else 0,
            'anomaly_score_std': float(np.std(anomaly_scores)) if len(anomaly_scores) > 0 else 0,
            'overall_score_mean': float(np.mean(scores)),
            'overall_score_std': float(np.std(scores))
        }
        
        return stats
    
    def print_detailed_stats(self, x_data, title="数据统计"):
        """
        以SSE格式打印详细统计数据
        
        Args:
            x_data: 数据
            title: 标题
        """
        stats = self.get_detailed_stats(x_data)
        sse_print("detailed_stats", {
            "title": title,
            "statistics": stats
        })


def demo_anomaly_detection():
    """
    演示异常检测功能
    """
    sse_print("start", {"message": "开始Isolation Forest投毒检测算法..."})
    
    # 获取数据参数
    param = {
        "dataset": "cifar10"
    }
    
    # 加载原始数据
    x_train, y_train, x_test, y_test = get_data(param)
    
    # 模拟一部分投毒数据
    num_poison = int(0.05 * x_train.shape[0])  # 5%的投毒比例
    x_train_mixed = x_train.copy()
    
    # 对一部分训练数据进行投毒（模拟BadNets攻击）
    for i in range(num_poison):
        # 在右下角添加触发器
        for c in range(3):
            for w in range(3):
                for h in range(3):
                    x_train_mixed[i][c][-(w+2)][-(h+2)] = 255
    
    sse_print("poisoned_samples_created", {
        "count": num_poison,
        "percentage": num_poison/x_train.shape[0]*100,
        "message": f"创建了 {num_poison} 个投毒样本 ({num_poison/x_train.shape[0]*100:.1f}%)"
    })
    
    # 初始化异常检测器
    detector = AnomalyDetector(contamination=0.1)
    
    # 使用大部分干净数据训练模型
    train_clean_data = x_train[num_poison:]
    detector.fit(train_clean_data)
    
    # 检测混合数据中的异常
    detector.evaluate_poisoning(
        x_clean=x_test[:1000],  # 使用测试集作为干净数据参考
        x_suspected_poisoned=x_train_mixed[:2000]  # 检测混合数据
    )
    
    # 输出详细统计数据
    detector.print_detailed_stats(x_test[:1000], "干净测试数据统计")
    detector.print_detailed_stats(x_train_mixed[:1000], "混合训练数据统计")
    
    sse_print("complete", {"message": "投毒防御完成"})
    return detector


if __name__ == "__main__":
    # 运行演示
    detector = demo_anomaly_detection()