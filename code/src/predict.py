import os
import multiprocessing as mp
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39
from enhanced_features import engineer_enhanced_features

feature_cloums_map = {
    '39': [
        'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
        'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
        'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ],
    '158+39': [
        'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
        'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
        'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
        'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
        'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
        'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
        'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
        'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
        'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
        'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
        'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
        'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
        'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
        'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
        'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
        'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
        'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ],
    'enhanced_158+39': None,
}

feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39,
    'enhanced_158+39': None,
}


def preprocess_predict_data(df, stockid2idx):
    feature_num = config['feature_num']

    if feature_num == 'enhanced_158+39':
        feature_engineer = engineer_features_158plus39
    elif feature_num in feature_engineer_func_map:
        feature_engineer = feature_engineer_func_map[feature_num]
    else:
        raise ValueError(f"Unsupported feature_num: {feature_num}")

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError('输入数据为空，无法预测')

    num_processes = min(10, mp.cpu_count())
    print(f'Using {num_processes} CPU cores')
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc='预测集特征工程'))

    processed = pd.concat(processed_list).reset_index(drop=True)

    if feature_num == 'enhanced_158+39':
        print("Applying enhanced cross-sectional features...")
        processed = engineer_enhanced_features(processed)

    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed['日期'] = pd.to_datetime(processed['日期'])

    if feature_num == 'enhanced_158+39':
        feature_columns = [c for c in processed.columns if c not in
                          ['股票代码', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额',
                           '振幅', '涨跌额', '换手率', '涨跌幅', 'open_t1', 'open_t5', 'label', 'instrument']]
    elif feature_num in feature_cloums_map:
        feature_columns = feature_cloums_map[feature_num]
    else:
        raise ValueError(f"Unknown feature_num: {feature_num}")

    return processed, feature_columns


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
    sequences, sequence_stock_ids = [], []
    for stock_id in stock_ids:
        stock_history = data[
            (data['股票代码'] == stock_id) &
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)

        if len(stock_history) == sequence_length:
            sequences.append(stock_history[features].values.astype(np.float32))
            sequence_stock_ids.append(stock_id)

    if len(sequences) == 0:
        raise ValueError('没有可用于预测的股票序列，请检查数据与 sequence_length')

    return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def allocate_weights(scores, stock_ids, top_k=5, method='confidence'):
    """
    Allocate portfolio weights based on model scores and confidence.

    method='confidence': weights proportional to softmax probability of top-k
    method='top_heavy': 50% #1, 25% #2, 15% #3, 10% #4
    method='equal': all stocks get equal weight 1.0/top_k
    """
    order = np.argsort(scores)[::-1]
    ranked_ids = [stock_ids[i] for i in order]
    ranked_scores = scores[order]

    top_ids = ranked_ids[:top_k]
    top_scores = ranked_scores[:top_k]

    if method == 'equal':
        w = 1.0 / len(top_ids)
        weights = [w] * len(top_ids)
    elif method == 'top_heavy':
        if len(top_ids) == 5:
            weights = [0.50, 0.25, 0.15, 0.10, 0.00]
        elif len(top_ids) == 4:
            weights = [0.55, 0.25, 0.13, 0.07]
        elif len(top_ids) == 3:
            weights = [0.60, 0.25, 0.15]
        elif len(top_ids) == 2:
            weights = [0.65, 0.35]
        else:
            weights = [1.0]
        weights = weights[:len(top_ids)]
    elif method == 'confidence':
        score_tensor = torch.FloatTensor(top_scores)
        probs = F.softmax(score_tensor / (config.get('score_temperature', 1.0)), dim=0).numpy()
        raw_weights = probs / probs.sum()
        weights = raw_weights.tolist()
        weights = [max(w, 0.05) for w in weights]
        total = sum(weights)
        weights = [w / total for w in weights]
    else:
        raise ValueError(f"Unknown weight method: {method}")

    return top_ids, weights


def main():
    data_file = os.path.join(config['data_path'], 'train.csv')
    model_path = os.path.join(config['output_dir'], 'best_model.pth')
    scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
    output_path = os.path.join('./output/', 'result.csv')
    os.makedirs('./output/', exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'未找到模型文件: {model_path}')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f'未找到Scaler文件: {scaler_path}')

    raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    latest_date = raw_df['日期'].max()

    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    processed, features = preprocess_predict_data(raw_df, stockid2idx)
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = joblib.load(scaler_path)
    processed[features] = scaler.transform(processed[features])

    sequence_length = config['sequence_length']
    sequences_np, sequence_stock_ids = build_inference_sequences(
        processed, features, sequence_length, stock_ids, latest_date)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
        scores = model(x).squeeze(0).detach().cpu().numpy()

    order = np.argsort(scores)[::-1]
    ranked_stock_ids = [sequence_stock_ids[i] for i in order]

    if len(ranked_stock_ids) < 5:
        raise ValueError(f'可预测股票不足5只，当前仅有 {len(ranked_stock_ids)} 只')

    print(f"预测日期: {latest_date.date()}")
    print(f"参与排序股票数: {len(ranked_stock_ids)}")

    ranked_scores = scores[order]
    print("Top 10 scores:")
    for i, (sid, sc) in enumerate(zip(ranked_stock_ids[:10], ranked_scores[:10])):
        print(f"  {i+1}. {sid}: {sc:.4f}")

    top_ids, weights = allocate_weights(ranked_scores, ranked_stock_ids, top_k=5, method='confidence')

    print("\nPortfolio allocation:")
    for sid, w in zip(top_ids, weights):
        print(f"  {sid}: {w:.4f}")
    print(f"  Total weight: {sum(weights):.4f}")

    output_df = pd.DataFrame({
        'stock_id': top_ids,
        'weight': weights,
    })
    output_df.to_csv(output_path, index=False, encoding='utf-8')

    print(f'\n结果已写入: {output_path}')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
