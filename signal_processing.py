import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

# 시리얼 연결 설정 (macOS 포트 이름으로 변경)
port = '/dev/cu.usbmodem11401'  # 실제 포트 이름으로 수정
baud_rate = 9600

try:
    arduino = serial.Serial(port, baud_rate, timeout=1)
    print(f"포트 {port}에 성공적으로 연결되었습니다.")
    time.sleep(2)  # 아두이노 연결 안정화 대기
except serial.SerialException as e:
    print(f"시리얼 연결 오류: {e}")
    available_ports = [p.device for p in serial.tools.list_ports.comports()]
    print(f"사용 가능한 포트: {available_ports}")
    exit(1)

# 데이터 수집 및 처리 함수
def collect_emg_data(duration=10, sampling_rate=100):
    data = []
    start_time = time.time()
    
    print(f"{duration}초 동안 데이터 수집 중...")
    
    while time.time() - start_time < duration:
        if arduino.in_waiting:
            try:
                line = arduino.readline().decode('utf-8').strip()
                value = float(line)
                data.append(value)
            except (ValueError, UnicodeDecodeError) as e:
                print(f"데이터 읽기 오류: {e}")
                continue
    
    print(f"{len(data)} 개의 데이터 포인트 수집 완료")
    return np.array(data)

# 신호 필터링 함수
def filter_emg_signal(data, sampling_rate=100):
    if len(data) == 0:
        print("경고: 필터링할 데이터가 없습니다.")
        return np.array([])
    
    print("필터링 시작...")
    
    # 대역 통과 필터 (20-500Hz)
    b, a = signal.butter(4, [20/(sampling_rate/2), 500/(sampling_rate/2)], 'bandpass')
    filtered_data = signal.filtfilt(b, a, data)
    
    # 노치 필터 (60Hz)
    b_notch, a_notch = signal.iirnotch(60, 30, sampling_rate)
    notched_data = signal.filtfilt(b_notch, a_notch, filtered_data)
    
    # 이동 평균 필터
    window_size = 5
    smoothed_data = np.convolve(notched_data, np.ones(window_size)/window_size, mode='valid')
    
    # 신호 정류화 (절대값)
    rectified_data = np.abs(smoothed_data)
    
    print("필터링 완료")
    return rectified_data

# 임계값 분석 함수
def analyze_threshold(data):
    if len(data) == 0:
        print("경고: 분석할 데이터가 없습니다.")
        return {'mean': 0, 'std': 0, 'thresholds': [0, 0, 0]}
    
    mean = np.mean(data)
    std = np.std(data)
    max_val = np.max(data)
    min_val = np.min(data)
    
    # 잠재적 임계값들 계산
    thresholds = [mean + i * std for i in range(1, 4)]
    
    return {
        'mean': mean,
        'std': std,
        'max': max_val,
        'min': min_val,
        'thresholds': thresholds
    }

# 데이터 저장 함수
def save_data(raw_data, filtered_data, threshold_info):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"emg_data_{timestamp}"
    
    # NumPy 배열로 저장
    np.savez(filename, 
             raw_data=raw_data, 
             filtered_data=filtered_data,
             mean=threshold_info['mean'],
             std=threshold_info['std'],
             thresholds=threshold_info['thresholds'])
    
    print(f"데이터가 '{filename}.npz'로 저장되었습니다.")

# 메인 실행 코드
if __name__ == "__main__":
    try:
        # 사용자에게 준비 시간 제공
        duration = 10
        print(f"EMG 데이터 수집을 시작합니다. {duration}초 동안 측정합니다.")
        print("준비가 되면 Enter 키를 누르세요...")
        input()
        
        print("EMG 데이터 수집 시작...")
        raw_data = collect_emg_data(duration=duration)
        
        if len(raw_data) > 0:
            print("신호 필터링 중...")
            filtered_data = filter_emg_signal(raw_data)
            
            print("임계값 분석 중...")
            threshold_info = analyze_threshold(filtered_data)
            
            # 결과 출력
            print("\n===== EMG 신호 분석 결과 =====")
            print(f"수집된 데이터 포인트: {len(raw_data)}")
            print(f"평균: {threshold_info['mean']:.2f}")
            print(f"표준편차: {threshold_info['std']:.2f}")
            print(f"최대값: {threshold_info['max']:.2f}")
            print(f"최소값: {threshold_info['min']:.2f}")
            print("\n잠재적 임계값:")
            for i, th in enumerate(threshold_info['thresholds']):
                print(f"  임계값 {i+1}: {th:.2f} (평균 + {i+1}σ)")
            
            # 데이터 저장
            save_data(raw_data, filtered_data, threshold_info)
            
            # 데이터 시각화
            plt.figure(figsize=(12, 8))
            
            # 원본 신호
            plt.subplot(2, 1, 1)
            plt.plot(raw_data, label='원본 신호')
            plt.title('EMG 원본 신호')
            plt.xlabel('샘플')
            plt.ylabel('신호 강도')
            plt.grid(True)
            plt.legend()
            
            # 필터링된 신호
            plt.subplot(2, 1, 2)
            plt.plot(np.arange(len(filtered_data)), filtered_data, label='필터링된 신호')
            
            for i, th in enumerate(threshold_info['thresholds']):
                plt.axhline(y=th, color=f'C{i+2}', linestyle='--', 
                            label=f'임계값 {i+1}: {th:.2f}')
            
            plt.title('EMG 필터링된 신호 및 임계값')
            plt.xlabel('샘플')
            plt.ylabel('신호 강도')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        else:
            print("데이터 수집 실패: 데이터가 수집되지 않았습니다.")
            
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        # 연결 종료
        arduino.close()
        print("시리얼 연결이 종료되었습니다.")
        