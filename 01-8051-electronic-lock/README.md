# 8051 单片机电子密码锁

课程设计（微处理器原理与接口综合实验）。AT89C51 + Keil C51 + Proteus，源码 426 行。

## 功能

| 功能 | 说明 |
|---|---|
| 密码输入 | 4×4 矩阵键盘输入 4 位密码，输入过程中数码管逐位显示 `8`，不显示真实数字 |
| 密码校验 | 按 `D` 确认后与存储的密码比对，正确则指示灯亮 1 秒 |
| 错误反馈 | 密码错误时数码管闪烁显示 `FFFF`，蜂鸣器响 1 声 |
| 连续三次错误锁定 | 蜂鸣器长鸣，键盘失效，只能复位重启 |
| 错误冷却 | 每次错误后 3 秒内不接受输入，中途按键会重新计时 |
| 输入超时报警 | 开始输入后 10 秒未确认，自动清空并蜂鸣 4 声 |
| 回删 | `#` 键退格，删除最近输入的一位 |
| 修改密码 | 验证通过后按 `D` 写入新密码 |
| 万用开锁 | `B` 键开锁，30 秒后自动上锁 |
| 复位 | `C` 键清空输入与显示 |

默认密码 `1234`，万能码 `9999`，都在 `src/CX.c` 里。

## 硬件与端口

| 端口 | 用途 |
|---|---|
| `P1` | 4×4 矩阵键盘 |
| `P3^0` | 指示灯（低电平点亮） |
| `P3^1` | 蜂鸣器 |
| `P2^0 ~ P2^3` | 4 位数码管位选 |
| `P0` | 数码管段码输出 |
| `T0` | 定时器 0，模式 1（16 位），1ms 中断 |

## 实现要点

### 1ms 定时器中断

```c
void Time0_init(void) {
  TMOD |= 0x01;                 // 定时器0，模式1（16位）
  TH0 = (65536-1000)/256;       // 1ms 初值
  TL0 = (65536-1000)%256;
  EA = 1; ET0 = 1; TR0 = 1;
}
```

中断里做两件事：维护几组软件计时器，以及刷新一位数码管。

```c
void time0(void) interrupt 1 {
  if (++temp_tiem_counter >= 100) { temp_tiem_counter = 0; Close_Lock_time++; }  // 100ms
  if (++ms_delay > 1000)                      ms_delay = 0;
  if (++LED_on_delay > 10000)                 LED_on_delay = 10000;
  if (++BUZZ_delay > 9999)                    BUZZ_delay = 10000;
  if (++password_wrong_delay > 9999)          password_wrong_delay = 10000;
  if (++Enter_password_outtime_delay > 20000) Enter_password_outtime_delay = 200000;
  Dis_data();
  TH0 = (65536-1000)/256; TL0 = (65536-1000)%256;
}
```

显示刷新放在中断里是为了让扫描周期稳定，不受主循环耗时影响；所有延时都用计数实现，主循环里没有 `delay()`。

### 矩阵键盘扫描

```c
uchar Key_scan(void) {
  static uchar key_down;
  Key_Port = 0xf0;                        // 行线输出低，列线输入
  if (Key_Port != 0xf0) {
    if (key_down == 0) {
      delay(10);                          // 消抖
      tem = Key_Port; Key_Port = 0xff;
      Key_Port = (tem | 0x0f);            // 行列反转，读列线
      key_num = Key_Port;
      key_down = 1;                       // 防止长按连发
      for (i = 0; i < 16; i++)
        if (keynum_tab[i] == key_num) break;
      return i;
    }
  } else key_down = 0;
  return 0xff;
}
```

### 数码管动态扫描

```c
void Dis_data(void) {
  static uchar temp = 0;
  P0 = 0x00;                              // 消隐
  switch (temp) {
    case 0: seg1=0;seg2=1;seg3=1;seg4=1; P0 = SEG_Table[Display_table[0]]; break;
    ...
  }
  if (++temp > 3) temp = 0;               // 每位 1ms，四位 4ms 一帧
}
```

### 状态标志

- `Right_password` / `unLock`：开锁状态
- `password_wrong_flag` + `password_wrong_delay`：错误后的 3 秒冷却
- `LED_Flashnum`：错误时 `FFFF` 的闪烁次数
- `BUZZ_on_num` + `BUZZ_delay`：蜂鸣器响几声
- `Enter_password_outtime_delay`：输入超时计时
- `Close_Lock_time`：万用开锁后的 30 秒计时

## 目录结构

```
01-8051-electronic-lock/
├── src/CX.c                     # 完整源码
├── keil/cx.uvproj               # Keil 工程
├── proteus/1.DSN                # Proteus 仿真原理图
├── proteus/1.PWI                # Proteus 仿真设置
└── firmware/electronic_lock.hex # 编译产物
```

原始 `CX.c` 是 GBK 编码（Keil 里的中文注释），这份转成了 UTF-8，代码内容没有改动。

## 编译与仿真

1. Proteus 打开 `proteus/1.DSN`，载入 hex 运行；
2. Keil 打开 `keil/cx.uvproj`，Rebuild 可重新生成 hex；
3. 实物部分是在实验箱上接线的。

## 已知问题

代码保持原样，没有修改。

| 位置 | 现象 |
|---|---|
| `main()` 三次错误分支 | `while(1){ Key_Port = 0xff; }` 死循环，锁定后无法自行恢复，必须硬件复位 |
| `#` 回删分支 | `Save_password_table[position--] = 0;` 中 `position` 是 `char`，在 `position==0` 时自减会下溢 |
| 密码校验 | 万能码 `9999` 硬编码在逻辑里，属于调试后门 |
| `Password_table[8]` | 初始化为 8 个元素，实际只用前 4 位 |
