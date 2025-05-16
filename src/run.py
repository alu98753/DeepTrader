import argparse
import json
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import copy
import time
from datetime import datetime
import logging
from tqdm import *

from torch.utils.tensorboard import SummaryWriter

from utils.parse_config import ConfigParser
from utils.functions import *
from agent import *
from environment.portfolio_env import PortfolioEnv


def run(func_args):
    if func_args.seed != -1:
        setup_seed(func_args.seed)

    # 在開始時檢查 PyTorch CUDA 狀態
    print(f"DEBUG: Initial device setting from args: {func_args.device}")
    if str(func_args.device) == 'cuda': # 比較字串形式
        if torch.cuda.is_available():
            print(f"DEBUG: PyTorch CUDA is available. Current device: {torch.cuda.current_device()}")
        else:
            print("DEBUG: WARNING - PyTorch CUDA is NOT available, but device is set to CUDA.")
            # 考慮是否要強制切換到 CPU 或報錯退出
            # func_args.device = torch.device('cpu')
            # print(f"DEBUG: Device overridden to CPU: {func_args.device}")

    data_prefix = './data/' + func_args.market + '/'
    matrix_path = data_prefix + func_args.relation_file

    start_time = datetime.now().strftime('%m%d/%H:%M:%S')
    if func_args.mode == 'train':
        PREFIX = 'outputs/'
        PREFIX = os.path.join(PREFIX, start_time)
        img_dir = os.path.join(PREFIX, 'img_file')
        save_dir = os.path.join(PREFIX, 'log_file')
        model_save_dir = os.path.join(PREFIX, 'model_file')

        print("DEBUG: About to create directories...")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)
        if not os.path.isdir(model_save_dir):
            os.mkdir(model_save_dir)
        print(f"DEBUG: Directories created/ensured at {PREFIX}")

        hyper_dict_copy = copy.deepcopy(func_args.__dict__) # 您的程式碼中已有 hyper 變數，這裡避免衝突
        # 您看到的 print(hyper) 在此之後
        # print(hyper_dict_copy) # 這行是您目前看到的輸出
        
        # 確保 device 是字串形式以便 JSON 序列化
        if isinstance(hyper_dict_copy['device'], torch.device):
            hyper_dict_copy['device'] = str(hyper_dict_copy['device'])
        
        json_str = json.dumps(hyper_dict_copy, indent=4)
        
        hyper_json_path = os.path.join(save_dir, 'hyper.json')
        print(f"DEBUG: About to write hyper.json to {hyper_json_path}")
        with open(hyper_json_path, 'w') as json_file:
            json_file.write(json_str)
        print("DEBUG: hyper.json written.")

        print(f"DEBUG: About to initialize SummaryWriter with log_dir: {save_dir}")
        writer = SummaryWriter(log_dir=save_dir) # TensorBoard 的 log_dir 參數
        print("DEBUG: SummaryWriter initialized.")
        # writer.add_text('hyper_setting', str(hyper_dict_copy)) # 您可以稍後再加回來
        # print("DEBUG: Hyper settings added to SummaryWriter.")

        print("DEBUG: About to set up logger...")
        logger = logging.getLogger()
        logger.setLevel('INFO') # 設定 logger 的INFO級別
        BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
        DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
        
        # 控制台 Handler
        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        chlr.setLevel(logging.INFO) # 將控制台級別也設為 INFO，這樣可以看到 logger.info
        
        # 檔案 Handler
        fhlr_path = os.path.join(save_dir, 'logger.log')
        print(f"DEBUG: FileHandler path: {fhlr_path}")
        fhlr = logging.FileHandler(fhlr_path)
        fhlr.setFormatter(formatter)
        fhlr.setLevel(logging.INFO) # 檔案日誌也設為 INFO
        
        # 清除已有的 handlers (如果重複運行此函數)
        if logger.hasHandlers():
            logger.handlers.clear()
            
        logger.addHandler(chlr)
        logger.addHandler(fhlr)
        print("DEBUG: Logger setup complete.")
        logger.info("DEBUG: Logger initialized - This is an INFO message.")
        logger.warning("DEBUG: Logger initialized - This is a WARNING message.")


        print(f"DEBUG: Loading data for market: {func_args.market}")
        # --- 資料載入 ---
        if func_args.market == 'DJIA':
            print("DEBUG: Loading DJIA stocks_data.npy...")
            stocks_data = np.load(data_prefix + 'stocks_data.npy')
            print(f"DEBUG: stocks_data.npy loaded. Shape: {stocks_data.shape}, Dtype: {stocks_data.dtype}")
            
            print("DEBUG: Loading DJIA ROR.npy...")
            rate_of_return = np.load( data_prefix + 'ROR.npy').astype(np.float32)
            print(f"DEBUG: ROR.npy loaded. Shape: {rate_of_return.shape}, Dtype: {rate_of_return.dtype}")
            
            print("DEBUG: Loading DJIA market_data.npy...")
            market_history = np.load(data_prefix + 'market_data.npy').astype(np.float32)
            print(f"DEBUG: market_data.npy loaded. Shape: {market_history.shape}, Dtype: {market_history.dtype}")
            
            assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size error'
            print("DEBUG: Shape assertion passed.")
            
            print(f"DEBUG: Loading matrix from {matrix_path}...")
            A_np = np.load(matrix_path).astype(np.float32)
            print(f"DEBUG: Matrix loaded from path. Shape: {A_np.shape}, Dtype: {A_np.dtype}")
            A = torch.from_numpy(A_np).float().to(func_args.device)
            print(f"DEBUG: Matrix A converted to tensor and moved to {func_args.device}. Shape: {A.shape}")
            
            test_idx = 7328
            allow_short = True
        # ... (為 HSI 和 CSI100 加入類似的 DEBUG print) ...
        elif func_args.market == 'HSI':
            # ... (加入 print)
            stocks_data = np.load(data_prefix + 'stocks_data.npy')
            print(f"DEBUG: HSI stocks_data.npy loaded. Shape: {stocks_data.shape}")
            rate_of_return = np.load(data_prefix + 'ROR.npy')
            print(f"DEBUG: HSI ROR.npy loaded. Shape: {rate_of_return.shape}")
            market_history = np.load(data_prefix + 'market_data.npy')
            print(f"DEBUG: HSI market_data.npy loaded. Shape: {market_history.shape}")
            assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size erROR'
            A_np = np.load(matrix_path)
            print(f"DEBUG: HSI Matrix loaded from path. Shape: {A_np.shape}")
            A = torch.from_numpy(A_np).float().to(func_args.device)
            print(f"DEBUG: HSI Matrix A converted to tensor and moved to {func_args.device}. Shape: {A.shape}")
            test_idx = 4211
            allow_short = True
        elif func_args.market == 'CSI100':
            # ... (加入 print)
            stocks_data = np.load(data_prefix + 'stocks_data.npy')
            print(f"DEBUG: CSI100 stocks_data.npy loaded. Shape: {stocks_data.shape}")
            rate_of_return = np.load(data_prefix + 'ROR.npy')
            print(f"DEBUG: CSI100 ROR.npy loaded. Shape: {rate_of_return.shape}")
            A_np = np.load(matrix_path)
            print(f"DEBUG: CSI100 Matrix loaded from path. Shape: {A_np.shape}")
            A = torch.from_numpy(A_np).float().to(func_args.device)
            print(f"DEBUG: CSI100 Matrix A converted to tensor and moved to {func_args.device}. Shape: {A.shape}")
            test_idx = 1944
            market_history = None
            allow_short = False
        else:
            print(f"FATAL: Market '{func_args.market}' not recognized in data loading section.")
            logger.error(f"Market '{func_args.market}' not recognized.")
            return

        print("DEBUG: Data loading complete.")
        print("DEBUG: About to initialize PortfolioEnv...")
        env = PortfolioEnv(assets_data=stocks_data, market_data=market_history, rtns_data=rate_of_return,
                           in_features=func_args.in_features, val_idx=test_idx, test_idx=test_idx,
                           batch_size=func_args.batch_size, window_len=func_args.window_len, trade_len=func_args.trade_len,
                           max_steps=func_args.max_steps, mode=func_args.mode, norm_type=func_args.norm_type,
                           allow_short=allow_short)
        print("DEBUG: PortfolioEnv initialized.")

        supports = [A]
        print(f"DEBUG: About to initialize RLActor with device: {func_args.device}...")
        actor = RLActor(supports, func_args).to(func_args.device)
        print("DEBUG: RLActor initialized.")
        
        print("DEBUG: About to initialize RLAgent...")
        agent = RLAgent(env, actor, func_args, logger=logger) # 將 logger 傳遞給 agent
        print("DEBUG: RLAgent initialized.")

        print("DEBUG: About to calculate mini_batch_num...")
        # 檢查 env.src.order_set 是否存在以及其長度
        if hasattr(env, 'src') and hasattr(env.src, 'order_set'):
            print(f"DEBUG: Length of env.src.order_set: {len(env.src.order_set)}")
            mini_batch_num = int(np.ceil(len(env.src.order_set) / func_args.batch_size))
            print(f"DEBUG: mini_batch_num calculated: {mini_batch_num}")
        else:
            print("DEBUG: ERROR - env.src.order_set not found or not initialized correctly!")
            logger.error("env.src.order_set not found or not initialized correctly!")
            return

        try:
            max_cr = 0
            print("DEBUG: Entering training loop...") # 您應該看到這個訊息，如果一切順利
            logger.info("DEBUG: Entering training loop...")
            for epoch in range(func_args.epochs):
                # logger.info(f'Epoch {epoch}/{func_args.epochs} starting...') # INFO 級別
                print(f'Epoch {epoch}/{func_args.epochs} starting...') # 直接 print

                epoch_return = 0
                for j in tqdm(range(mini_batch_num)):
                    episode_return, avg_rho, avg_mdd = agent.train_episode()
                    epoch_return += episode_return
                avg_train_return = epoch_return / mini_batch_num
                logger.warning('[%s]round %d, avg train return %.4f, avg rho %.4f, avg mdd %.4f' %
                               (start_time, epoch, avg_train_return, avg_rho, avg_mdd))
                agent_wealth = agent.evaluation()
                metrics = calculate_metrics(agent_wealth, func_args.trade_mode)
                writer.add_scalar('Test/APR', metrics['APR'], global_step=epoch)
                writer.add_scalar('Test/MDD', metrics['MDD'], global_step=epoch)
                writer.add_scalar('Test/AVOL', metrics['AVOL'], global_step=epoch)
                writer.add_scalar('Test/ASR', metrics['ASR'], global_step=epoch)
                writer.add_scalar('Test/SoR', metrics['DDR'], global_step=epoch)
                writer.add_scalar('Test/CR', metrics['CR'], global_step=epoch)

                if metrics['CR'] > max_cr:
                    print('New Best CR Policy!!!!')
                    max_cr = metrics['CR']
                    torch.save(actor, os.path.join(model_save_dir, 'best_cr-'+str(epoch)+'.pkl'))
                logger.warning('after training %d round, max wealth: %.4f, min wealth: %.4f,'
                               ' avg wealth: %.4f, final wealth: %.4f, ARR: %.3f%%, ASR: %.3f, AVol" %.3f,'
                               'MDD: %.2f%%, CR: %.3f, DDR: %.3f'
                               % (
                                   epoch, max(agent_wealth[0]), min(agent_wealth[0]), np.mean(agent_wealth),
                                   agent_wealth[-1, -1], 100 * metrics['APR'], metrics['ASR'], metrics['AVOL'],
                                   100 * metrics['MDD'], metrics['CR'], metrics['DDR']
                               ))
        except KeyboardInterrupt:
            torch.save(actor, os.path.join(model_save_dir, 'final_model.pkl'))
            torch.save(agent.optimizer.state_dict(), os.path.join(model_save_dir, 'final_optimizer.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('--window_len', type=int)
    parser.add_argument('--G', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--no_spatial', dest='spatial_bool', action='store_false')
    parser.add_argument('--no_msu', dest='msu_bool', action='store_false')
    parser.add_argument('--relation_file', type=str)
    parser.add_argument('--addaptiveadj', dest='addaptive_adj_bool', action='store_false')

    opts = parser.parse_args()

    if opts.config is not None:
        with open(opts.config) as f:
            options = json.load(f)
            args = ConfigParser(options)
    else:
        with open('./hyper.json') as f:
            options = json.load(f)
            args = ConfigParser(options)
    args.update(opts)

    run(args)
