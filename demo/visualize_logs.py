import os
import re
import sys
import glob

def find_latest_log(log_dir="logs"):
    if not os.path.exists(log_dir):
        return None
    logs = glob.glob(os.path.join(log_dir, "*.log"))
    if not logs: 
        return None
    # 返回最新修改的日志
    return max(logs, key=os.path.getmtime)

def parse_log(filepath):
    steps = []
    current_step = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        text = line.strip()
        m_step = re.match(r'^---\s*Step\s+(\d+)\s*---', text)
        if m_step:
            if current_step:
                steps.append(current_step)
            current_step = {
                "step": int(m_step.group(1)),
                "thought": "",
                "action": "",
                "success": False,
                "feedback": "",
                "warnings": []
            }
            continue
            
        if not current_step:
            continue
            
        if text.startswith("<thought>"):
            content = text.split(":", 1)[1].strip() if ":" in text else text
            current_step["thought"] = content
        elif text.startswith("<action>"):
            content = text.split(":", 1)[1].strip() if ":" in text else text
            current_step["action"] = content
        elif text.startswith("Env feedback:"):
            # e.g. [✓] 动作成功
            content = text.split(":", 1)[1].strip() if ":" in text else text
            current_step["feedback"] = content
            if "[✓]" in text:
                current_step["success"] = True
            else:
                current_step["success"] = False
        elif text.startswith("└─ [WARNING]"):
            warn_msg = text.replace("└─ [WARNING]", "").strip()
            current_step["warnings"].append(warn_msg)
            
    if current_step:
        steps.append(current_step)
        
    return steps

def generate_html(steps, output_path, title):
    total_steps = len(steps)
    success_steps = sum(1 for s in steps if s['success'])
    total_warnings = sum(len(s['warnings']) for s in steps)
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>NS-FSM 动态轨迹重放</title>
    <style>
        body {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #f4f6f9; color: #333; padding: 30px; line-height: 1.6; margin: 0; }}
        h1 {{ text-align: center; color: #2c3e50; font-size: 2.2em; margin-bottom: 5px; }}
        .subtitle {{ text-align: center; color: #7f8c8d; margin-bottom: 30px; font-size: 1.1em; }}
        
        .controls {{ text-align: center; position: sticky; top: 15px; z-index: 1000; margin-bottom: 30px; }}
        .btn {{
            padding: 12px 25px; font-size: 1.1em; background: #3498db; color: white; border: none; 
            border-radius: 8px; cursor: pointer; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.2s;
            margin: 0 10px; font-weight: bold;
        }}
        .btn:hover {{ background: #2980b9; transform: translateY(-2px); }}
        .btn-outline {{ background: white; color: #3498db; border: 2px solid #3498db; }}
        .btn-outline:hover {{ background: #f0f8ff; }}
        
        .stats-panel {{ 
            max-width: 800px; margin: 0 auto 40px auto; background: white; padding: 25px; 
            border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); 
            display: flex; justify-content: space-around; border-top: 5px solid #3498db;
        }}
        .stat-box {{ text-align: center; }}
        .stat-num {{ display: block; font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .stat-label {{ color: #7f8c8d; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }}
        
        .timeline-container {{ 
            max-width: 800px; margin: 0 auto; padding-left: 30px; 
            border-left: 4px solid #e0e6ed; position: relative;
        }}
        
        /* 动画初始状态：隐藏并在偏下方 */
        .step-card {{ 
            background: white; border-radius: 10px; padding: 20px; margin-bottom: 25px; 
            position: relative; box-shadow: 0 3px 10px rgba(0,0,0,0.04);
            border-left: 6px solid #e0e6ed; 
            opacity: 0; transform: translateY(40px); display: none;
            transition: opacity 0.6s ease-out, transform 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }}
        
        /* 动画触发状态：浮现显示 */
        .step-card.visible {{
            display: block; opacity: 1; transform: translateY(0);
        }}
        /* 提供一个瞬间展示的类 */
        .step-card.show-all {{
            display: block; opacity: 1; transform: translateY(0); transition: none;
        }}
        
        .step-card::before {{
            content: ''; position: absolute; left: -41px; top: 20px; 
            width: 18px; height: 18px; border-radius: 50%; background: #e0e6ed;
            border: 4px solid #f4f6f9;
        }}
        
        .success {{ border-left-color: #2ecc71; }}
        .success::before {{ background: #2ecc71; box-shadow: 0 0 0 3px rgba(46, 204, 113, 0.2); }}
        .fail {{ border-left-color: #e74c3c; }}
        .fail::before {{ background: #e74c3c; box-shadow: 0 0 0 3px rgba(231, 76, 60, 0.2); }}
        
        .step-num {{ font-weight: 800; color: #95a5a6; font-size: 1em; letter-spacing: 1px; margin-bottom: 12px; display: block; }}
        
        /* Thought 气泡样式 */
        .thought-box {{
            background: #fdfdfd; padding: 12px 15px; border-radius: 8px; font-style: italic; color: #626567;
            margin-bottom: 15px; border-left: 3px solid #f39c12; font-size: 0.95em; position: relative;
        }}
        .thought-box strong {{ font-style: normal; color: #d35400; }}
        
        .action-title {{ font-size: 1.4em; font-weight: bold; color: #2c3e50; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }}
        .action-tag {{ font-size: 0.6em; background: #34495e; color: white; padding: 3px 8px; border-radius: 4px; vertical-align: top; font-weight: normal; letter-spacing: 1px; }}
        
        .feedback {{ font-size: 1em; color: #444; background: #f8f9fa; padding: 12px; border-radius: 6px; border: 1px solid #eee; }}
        
        .warnings {{ margin-top: 15px; display: flex; flex-direction: column; gap: 8px; }}
        .warning-tag {{ 
            background: #fff8e1; color: #e67e22; font-size: 0.95em; padding: 8px 12px; 
            border-radius: 6px; border: 1px solid #ffeaa7; font-weight: 600;
            display: flex; align-items: center; gap: 8px;
        }}
        .warning-tag::before {{ content: '⚠️'; font-size: 1.1em; }}
        
    </style>
</head>
<body>
    <h1>🎬 NS-FSM: 实验轨迹动态重放</h1>
    <div class="subtitle">{title}</div>
    
    <div class="controls">
        <button class="btn" id="playBtn" onclick="togglePlay()">▶ 播放实验回放</button>
        <button class="btn btn-outline" onclick="showAll()">👁 展开全部步骤</button>
    </div>
    
    <div class="stats-panel">
        <div class="stat-box">
            <span class="stat-num">{total_steps}</span>
            <span class="stat-label">总步骤数</span>
        </div>
        <div class="stat-box">
            <span class="stat-num" style="color: #2ecc71;">{success_steps}</span>
            <span class="stat-label">有效推进</span>
        </div>
        <div class="stat-box">
            <span class="stat-num" style="color: #e74c3c;">{total_steps - success_steps}</span>
            <span class="stat-label">撞墙/错误</span>
        </div>
        <div class="stat-box">
            <span class="stat-num" style="color: #f39c12;">{total_warnings}</span>
            <span class="stat-label">Lost In Mid 异常</span>
        </div>
    </div>
    
    <div class="timeline-container" id="timeline">
"""

    for s in steps:
        status_class = "success" if s['success'] else "fail"
        action_text = s['action'] if s['action'] else "UNKNOWN ACTION"
        feedback = s['feedback'] if s['feedback'] else "无反馈记录"
        thought_html = f"<div class='thought-box'><strong>🤔 内部思绪 (Thought):</strong><br>{s['thought']}</div>" if s.get('thought') else ""
        
        warn_html = ""
        if s['warnings']:
            warn_html += "<div class='warnings'>"
            for w in s['warnings']:
                warn_html += f"<div class='warning-tag'>{w}</div>"
            warn_html += "</div>"
            
        card = f"""
        <div class="step-card {status_class}">
            <span class="step-num">STEP {s['step']}</span>
            {thought_html}
            <div class="action-title"><span class="action-tag">ACTION</span> {action_text}</div>
            <div class="feedback">{feedback}</div>
            {warn_html}
        </div>
        """
        html += card
        
    html += """
    </div>
    <div style="height: 300px;"></div>
    
    <script>
        const cards = document.querySelectorAll('.step-card');
        const playBtn = document.getElementById('playBtn');
        let currentIndex = 0;
        let playInterval = null;
        let isPlaying = false;

        function showAll() {
            clearInterval(playInterval);
            isPlaying = false;
            playBtn.innerText = '↺ 重新回放';
            cards.forEach(card => card.classList.add('show-all'));
        }

        function togglePlay() {
            if (isPlaying) {
                // 暂停
                clearInterval(playInterval);
                playBtn.innerText = '▶ 继续回放';
                isPlaying = false;
            } else {
                // 确保移除全展示效果
                if (currentIndex >= cards.length || document.querySelector('.show-all')) {
                    cards.forEach(card => {
                        card.classList.remove('visible', 'show-all');
                        card.style.display = 'none';
                    });
                    currentIndex = 0;
                }
                
                playBtn.innerText = '⏸ 暂停回放';
                isPlaying = true;
                
                playInterval = setInterval(() => {
                    if (currentIndex < cards.length) {
                        const card = cards[currentIndex];
                        card.style.display = 'block'; // 先给 display: block
                        
                        // 强制触发重排，让 transition 生效
                        void card.offsetWidth; 
                        
                        card.classList.add('visible'); // 再加透明度变换和弹跳动画
                        
                        // 平滑滚动，留一点余量在顶部
                        const y = card.getBoundingClientRect().top + window.scrollY - 100;
                        window.scrollTo({ top: y, behavior: 'smooth' });
                        
                        currentIndex++;
                    } else {
                        clearInterval(playInterval);
                        playBtn.innerText = '↺ 重新回放';
                        isPlaying = false;
                    }
                }, 1200); // ✅ 每步停留 1.2 秒，您可以调整这个速度
            }
        }
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
        
    return output_path

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else find_latest_log()
        
    if not log_file:
        print("错误：未能在 logs/ 目录下找到任何 .log 文件！")
        sys.exit(1)
        
    print(f"解析动画素材: {log_file}")
    steps = parse_log(log_file)
    
    if not steps:
        print("解析异常，无步骤数据。")
        sys.exit(1)
        
    out_name = os.path.basename(log_file).replace(".log", "_anime.html")
    out_path = os.path.join(os.path.dirname(log_file), out_name)
    
    generate_html(steps, out_path, os.path.basename(log_file))
    print(f"\\n✅ 动画版的轨迹追踪器已保存到: {out_path}")
