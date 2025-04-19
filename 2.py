import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import pickle
from pathlib import Path
import matplotlib.font_manager as fm
from matplotlib import rcParams

# 한글 폰트 설정 (필요시 사용)
# plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

def load_results(results_dir):
    """결과 디렉토리에서 모든 결과 파일을 로드합니다."""
    results = {}
    
    # 요약 JSON 파일 로드
    summary_file = os.path.join(results_dir, "A_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            results["summary"] = json.load(f)
    
    # 각 에이전트별 결과 파일 로드
    for agent_type in ["EGreedy", "UCB", "TS", "CDQL", "ECIA"]:
        results_file = os.path.join(results_dir, f"A_{agent_type}_results.pkl")
        if os.path.exists(results_file):
            with open(results_file, "rb") as f:
                results[agent_type] = pickle.load(f)
    
    return results

def generate_statistics_tables(results):
    """통계 테이블 생성"""
    summary = results["summary"]
    
    # 표 1: 환경 변화 후 평균 보상
    post_change_rewards = summary["post_change_rewards"]
    table1 = pd.DataFrame({
        "Agent": list(post_change_rewards.keys()),
        "Mean Reward": [post_change_rewards[agent]["mean"] for agent in post_change_rewards],
        "Std Dev": [post_change_rewards[agent]["std"] for agent in post_change_rewards],
        "Sample Size": [post_change_rewards[agent]["sample_size"] for agent in post_change_rewards]
    })
    
    # 표 2: 회복 시간
    recovery_times = summary["recovery_times"]
    table2 = pd.DataFrame({
        "Agent": list(recovery_times.keys()),
        "Mean Recovery Time": [recovery_times[agent]["mean"] for agent in recovery_times],
        "Std Dev": [recovery_times[agent]["std"] for agent in recovery_times],
        "Median": [recovery_times[agent]["median"] for agent in recovery_times],
        "Sample Size": [recovery_times[agent]["sample_size"] for agent in recovery_times]
    })
    
    # 표 3: 회복률 분석 (10회, 50회)
    recovery_analysis = summary["stats_results"]["recovery_analysis"]
    
    # 10회 회복률
    recovery_10 = {agent: recovery_analysis["recovery_after_10"][agent] 
                  for agent in ["EGreedy", "UCB", "TS", "CDQL", "ECIA"]}
    table3a = pd.DataFrame({
        "Agent": list(recovery_10.keys()),
        "Mean Recovery Rate (10 trials)": [recovery_10[agent]["mean_recovery_rate"] for agent in recovery_10],
        "Std Dev": [recovery_10[agent]["std_recovery_rate"] for agent in recovery_10],
        "Sample Size": [recovery_10[agent]["sample_size"] for agent in recovery_10]
    })
    
    # 50회 회복률
    recovery_50 = {agent: recovery_analysis["recovery_after_50"][agent] 
                  for agent in ["EGreedy", "UCB", "TS", "CDQL", "ECIA"]}
    table3b = pd.DataFrame({
        "Agent": list(recovery_50.keys()),
        "Mean Recovery Rate (50 trials)": [recovery_50[agent]["mean_recovery_rate"] for agent in recovery_50],
        "Std Dev": [recovery_50[agent]["std_recovery_rate"] for agent in recovery_50],
        "Sample Size": [recovery_50[agent]["sample_size"] for agent in recovery_50]
    })
    
    # 표 4: 통계적 유의성 검정 결과
    post_change_stats = summary["stats_results"]["post_change_performance"]
    table4 = pd.DataFrame({
        "Comparison": list(post_change_stats.keys()),
        "t-statistic": [post_change_stats[comp]["t_statistic"] for comp in post_change_stats],
        "p-value": [post_change_stats[comp]["p_value"] for comp in post_change_stats],
        "Cohen's d": [post_change_stats[comp]["cohen_d"] for comp in post_change_stats],
        "Mean Difference": [post_change_stats[comp]["means"][0] - post_change_stats[comp]["means"][1] 
                           for comp in post_change_stats]
    })
    
    # 표 5: 감정 동태 분석
    emotion_dynamics = summary["emotion_dynamics"]
    table5 = pd.DataFrame({
        "Emotion": list(emotion_dynamics.keys()),
        "Pre-Change Mean": [emotion_dynamics[emo]["pre_change_mean"] for emo in emotion_dynamics],
        "Post-Change Mean": [emotion_dynamics[emo]["post_change_mean"] for emo in emotion_dynamics],
        "Adapted Mean": [emotion_dynamics[emo]["adapted_mean"] for emo in emotion_dynamics],
        "Change Ratio": [emotion_dynamics[emo]["change_ratio"] for emo in emotion_dynamics]
    })
    
    return {
        "post_change_rewards": table1,
        "recovery_times": table2,
        "recovery_rate_10": table3a,
        "recovery_rate_50": table3b,
        "statistical_tests": table4,
        "emotion_dynamics": table5
    }

def generate_latex_tables(tables):
    """LaTeX 형식의 테이블 생성"""
    latex_tables = {}
    
    for name, table in tables.items():
        latex_table = table.to_latex(index=False, float_format="%.4f")
        
        # 테이블 제목과 라벨 추가
        if name == "post_change_rewards":
            title = "환경 변화 후 평균 보상 (시행 100-199)"
            label = "tab:post_change_rewards"
        elif name == "recovery_times":
            title = "환경 변화 후 회복 시간 (시행 수)"
            label = "tab:recovery_times"
        elif name == "recovery_rate_10":
            title = "환경 변화 후 10회 시행 시점의 회복률"
            label = "tab:recovery_rate_10"
        elif name == "recovery_rate_50":
            title = "환경 변화 후 50회 시행 시점의 회복률"
            label = "tab:recovery_rate_50"
        elif name == "statistical_tests":
            title = "ECIA와 다른 알고리즘 간의 통계적 비교 (환경 변화 후)"
            label = "tab:statistical_tests"
        elif name == "emotion_dynamics":
            title = "ECIA의 감정 동태 분석"
            label = "tab:emotion_dynamics"
        
        latex_table = f"\\begin{{table}}[ht]\n\\centering\n\\caption{{{title}}}\n\\label{{{label}}}\n{latex_table}\n\\end{{table}}"
        latex_tables[name] = latex_table
    
    return latex_tables

def generate_figures(results, output_dir):
    """논문용 그래프 생성"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 기존 이미지 파일 복사 또는 새로 생성
    # (필요한 경우 여기서 이미지를 사용자 지정 형식으로 다시 생성할 수 있음)
    
    # 필요시 추가 분석 그래프 생성
    # 예: ECIA와 다른 알고리즘 간의 직접 비교 그래프
    
    # 환경 변화 전후 성능 비교 바 차트
    summary = results["summary"]
    post_change_rewards = summary["post_change_rewards"]
    
    agents = list(post_change_rewards.keys())
    means = [post_change_rewards[agent]["mean"] for agent in agents]
    stds = [post_change_rewards[agent]["std"] for agent in agents]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(agents, means, yerr=stds, alpha=0.7)
    
    # 높이 순서로 색상 지정
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    sorted_indices = np.argsort(means)
    for i, idx in enumerate(sorted_indices):
        bars[idx].set_color(colors[i])
    
    plt.title("Mean Reward After Environmental Change (Trials 100-199)")
    plt.xlabel("Agent")
    plt.ylabel("Mean Reward")
    plt.ylim(0, 0.6)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, "post_change_performance_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 회복 시간 비교 바 차트 (로그 스케일)
    recovery_times = summary["recovery_times"]
    agents = list(recovery_times.keys())
    means = [recovery_times[agent]["mean"] for agent in agents]
    stds = [recovery_times[agent]["std"] for agent in agents]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(agents, means, yerr=stds, alpha=0.7)
    
    # 값이 작을수록 좋은 것이므로 높이 역순으로 색상 지정
    sorted_indices = np.argsort([-m for m in means])
    for i, idx in enumerate(sorted_indices):
        bars[idx].set_color(colors[i])
    
    plt.title("Average Recovery Time After Environmental Change")
    plt.xlabel("Agent")
    plt.ylabel("Recovery Time (Trials)")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, "recovery_time_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 회복률 비교 (10회 vs 50회)
    recovery_analysis = summary["stats_results"]["recovery_analysis"]
    recovery_10 = {agent: recovery_analysis["recovery_after_10"][agent]["mean_recovery_rate"] 
                  for agent in ["EGreedy", "UCB", "TS", "CDQL", "ECIA"]}
    recovery_50 = {agent: recovery_analysis["recovery_after_50"][agent]["mean_recovery_rate"]
                  for agent in ["EGreedy", "UCB", "TS", "CDQL", "ECIA"]}
    
    agents = list(recovery_10.keys())
    r10_values = [recovery_10[agent] for agent in agents]
    r50_values = [recovery_50[agent] for agent in agents]
    
    x = np.arange(len(agents))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, r10_values, width, label='After 10 trials', alpha=0.7)
    plt.bar(x + width/2, r50_values, width, label='After 50 trials', alpha=0.7)
    
    plt.title("Recovery Rate Comparison (10 vs 50 trials post-change)")
    plt.xlabel("Agent")
    plt.ylabel("Recovery Rate")
    plt.xticks(x, agents)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, "recovery_rate_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    return "그래프가 성공적으로 생성되었습니다."

def main():
    # 결과 디렉토리 설정
    results_dir = "C:/Users/Administrator/OneDrive/내 논문들/인공지능이 의식을 가지게 하는 방법/통합 버전/Peerj/results_A/analysis_20250415_115402"
    output_dir = "C:/Users/Administrator/OneDrive/내 논문들/인공지능이 의식을 가지게 하는 방법/통합 버전/Peerj/paper_statistics"
    
    # 결과 로드
    results = load_results(results_dir)
    
    # 통계 테이블 생성
    tables = generate_statistics_tables(results)
    
    # LaTeX 테이블 생성
    latex_tables = generate_latex_tables(tables)
    
    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    
    # 일반 테이블 저장 (CSV)
    for name, table in tables.items():
        table.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
    
    # LaTeX 테이블 저장
    for name, latex_table in latex_tables.items():
        with open(os.path.join(output_dir, f"{name}.tex"), "w", encoding="utf-8") as f:
            f.write(latex_table)
    
    # 그래프 생성
    generate_figures(results, output_dir)
    
    # 요약 정보 추출
    summary = results["summary"]
    
    # 핵심 통계 요약 텍스트 파일 생성
    with open(os.path.join(output_dir, "key_statistics.txt"), "w", encoding="utf-8") as f:
        f.write("# ECIA 연구 핵심 통계 요약\n\n")
        
        f.write("## 1. 환경 변화 후 평균 보상\n")
        for agent, data in summary["post_change_rewards"].items():
            f.write(f"- {agent}: {data['mean']:.4f} ± {data['std']:.4f}\n")
        
        f.write("\n## 2. 평균 회복 시간\n")
        for agent, data in summary["recovery_times"].items():
            f.write(f"- {agent}: {data['mean']:.2f} ± {data['std']:.2f} (중앙값: {data['median']})\n")
        
        f.write("\n## 3. ECIA와 다른 알고리즘 비교 통계적 유의성\n")
        for comp, data in summary["stats_results"]["post_change_performance"].items():
            f.write(f"- {comp}: t = {data['t_statistic']:.2f}, p = {data['p_value']:.4e}, Cohen's d = {data['cohen_d']:.2f}\n")
        
        f.write("\n## 4. 회복률 (10회 시행 후)\n")
        for agent, data in summary["stats_results"]["recovery_analysis"]["recovery_after_10"].items():
            if isinstance(data, dict) and "mean_recovery_rate" in data:
                f.write(f"- {agent}: {data['mean_recovery_rate']:.4f} ± {data['std_recovery_rate']:.4f}\n")
        
        f.write("\n## 5. 회복률 (50회 시행 후)\n")
        for agent, data in summary["stats_results"]["recovery_analysis"]["recovery_after_50"].items():
            if isinstance(data, dict) and "mean_recovery_rate" in data:
                f.write(f"- {agent}: {data['mean_recovery_rate']:.4f} ± {data['std_recovery_rate']:.4f}\n")
        
        f.write("\n## 6. ECIA 감정 동태 (환경 변화 전 → 직후 → 적응 후)\n")
        for emotion, data in summary["emotion_dynamics"].items():
            f.write(f"- {emotion}: {data['pre_change_mean']:.4f} → {data['post_change_mean']:.4f} → {data['adapted_mean']:.4f}\n")
    
    print(f"분석 완료. 결과가 {output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()