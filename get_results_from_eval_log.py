
def get_results_from_log():
    path = "../test/eval_log.txt"
    
    with open(path, 'r') as f:
        log_content = f.read()
    
    win_count = 0
    lose_count = 0
    total_hp_diff = 0
    hp_diff = 0

    for line in log_content.split('\n'):
        if 'win' in line:
            win_count += 1
        elif 'lose' in line:
            lose_count += 1
            
        if 'At the end' in line:
            hp_diff = int(line.split(':')[1].split('.')[0].split()[-1]) - int(line.split(':')[0].split()[-1])
        
        total_hp_diff += hp_diff

    total_count = win_count + lose_count  # 总场次
    win_rate = win_count / total_count  # 胜率
    avg_hp_diff = total_hp_diff / total_count  # 平均血量差

    print(f"总场次: {total_count}")
    print(f"胜利场次: {win_count}, 失败场次: {lose_count}")
    print(f"胜率: {win_rate:.2f}")
    print(f"平均血量差: {avg_hp_diff:.2f}")

if __name__ == "__main__":
    get_results_from_log()