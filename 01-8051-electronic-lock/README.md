# 8051 单片机电子密码锁（防盗自动报警）

> 课程设计 · 微处理器原理与接口综合实验 · AT89C51 · Keil C51 · Proteus
> 一个用 **定时器中断做全系统时基** 的完整状态机实现：矩阵键盘输入、数码管动态扫描、密码校验与修改、超时报警、连续错误锁定。

---

## 1. 功能

| 功能 | 说明 |
|---|---|
| 密码输入 | 4×4 矩阵键盘输入 4 位密码，输入过程中数码管**逐位显示 `8` 做遮蔽**，不显示真实数字 |
| 密码校验 | 按 `D` 确认后与存储密码比对，正确则绿灯亮 1 秒 |
| 错误反馈 | 错误时数码管闪烁显示 `FFFF`（ERROR），蜂鸣器响 1 声 |
| **连续三次错误 → 锁定** | 蜂鸣器长鸣报警、键盘彻底失效，**只能重启复位解锁** |
| 错误冷却 | 每次错误后 **3 秒内不接受新输入**，中途按键会重新计时 |
| 输入超时报警 | 开始输入后 **10 秒未确认**，自动清空输入并蜂鸣 4 声报警 |
| 回删 | `#` 键退格，删除最近输入的一位 |
| 修改密码 | 验证通过（开锁状态）后按 `D` 即可写入新密码 |
| 万用开锁 | `B` 键万用开锁，开锁后 **30 秒自动重新上锁** |
| 复位 | `C` 键清空输入与显示 |

**默认密码 `1234`，万能码 `9999`**（见 `src/CX.c` 中的 `Password_table` 与校验逻辑）。

## 2. 硬件与端口分配

| 端口 | 用途 |
|---|---|
| `P1` | 4×4 矩阵键盘（行列反转扫描） |
| `P3^0` | LED 指示灯（低电平点亮） |
| `P3^1` | 蜂鸣器 |
| `P2^0 ~ P2^3` | 4 位数码管位选（`seg1~seg4`） |
| `P0` | 数码管段码输出（`SEG_Table`：0-F + 熄灭） |
| `T0` | 定时器 0，**模式 1（16 位）**，1ms 中断作为全系统时基 |

## 3. 关键实现

### 3.1 1ms 定时器中断 —— 全系统时基

```c
void Time0_init(void) {
  TMOD |= 0x01;                 // 定时器0，模式1（16位）
  TH0 = (65536-1000)/256;       // 1ms 初值
  TL0 = (65536-1000)%256;
  EA = 1; ET0 = 1; TR0 = 1;     // 开总中断 / T0 中断 / 启动
}
```

中断服务程序里用**多组软件计时器**并行管理所有时间相关逻辑，主循环只做按键与状态判断：

```c
void time0(void) interrupt 1 {
  if (++temp_tiem_counter >= 100) { temp_tiem_counter = 0; Close_Lock_time++; }  // 100ms 基准
  if (++ms_delay > 1000)                  ms_delay = 0;
  if (++LED_on_delay > 10000)             LED_on_delay = 10000;
  if (++BUZZ_delay > 9999)                BUZZ_delay = 10000;
  if (++password_wrong_delay > 9999)      password_wrong_delay = 10000;
  if (++Enter_password_outtime_delay > 20000) Enter_password_outtime_delay = 200000;
  Dis_data();                             // 1ms 刷新一位数码管（动态扫描）
  TH0 = (65536-1000)/256; TL0 = (65536-1000)%256;   // 重装初值
}
```

> 设计要点：**显示刷新放在中断里**，保证数码管扫描周期稳定、不因主循环耗时抖动而闪烁；所有延时都是**非阻塞计数**，没有 `delay()` 阻塞主循环。

### 3.2 矩阵键盘：行列反转扫描 + 软件消抖 + 防连发

```c
uchar Key_scan(void) {
  static uchar key_down;
  Key_Port = 0xf0;                        // 行线输出低、列线输入
  if (Key_Port != 0xf0) {                 // 有按键按下
    if (key_down == 0) {
      delay(10);                          // 延时 10ms 软件消抖
      tem = Key_Port; Key_Port = 0xff;
      Key_Port = (tem | 0x0f);            // 行列反转，读列线
      key_num = Key_Port;
      key_down = 1;                       // 静态变量防止长按连发
      for (i = 0; i < 16; i++)
        if (keynum_tab[i] == key_num) break;
      return i;                           // 返回键值 0~15
    }
  } else key_down = 0;
  return 0xff;                            // 无按键
}
```

### 3.3 数码管动态扫描（1ms 一位，4ms 一帧）

```c
void Dis_data(void) {
  static uchar temp = 0;
  P0 = 0x00;                              // 先消隐，避免拖影
  switch (temp) {
    case 0: seg1=0;seg2=1;seg3=1;seg4=1; P0 = SEG_Table[Display_table[0]]; break;
    ...
  }
  if (++temp > 3) temp = 0;               // 4 位循环
}
```

### 3.4 状态机与标志位

主循环是"**按键事件 → 修改状态 → 中断按时间为状态服务**"的结构，用标志位管理互斥状态：

- `Right_password` / `unLock`：开锁状态
- `password_wrong_flag` + `password_wrong_delay`：错误后的 3 秒冷却
- `LED_Flashnum`：错误时 `FFFF` 闪烁次数（每 500ms 翻转一次）
- `BUZZ_on_num` + `BUZZ_delay`：蜂鸣器响几声（隔 200ms 切换）
- `Enter_password_outtime_delay`：输入超时（>9999 即 10s）
- `Close_Lock_time`：万用开锁后 30 秒自动上锁

## 4. 目录结构

```
01-8051-electronic-lock/
├── src/CX.c                    # 完整源码（426 行，含中文注释）
├── keil/cx.uvproj              # Keil µVision 工程文件
├── proteus/1.DSN               # Proteus 仿真原理图
├── proteus/1.PWI               # Proteus 仿真设置
└── firmware/electronic_lock.hex # 编译产物，可直接烧录到 AT89C51
```

> **编码说明**：原始 `CX.c` 为 GBK 编码（Keil 中文注释），仓库内已转为 UTF-8，代码内容未做任何修改，仅编码转换。

> 📌 完整课程设计报告（含原理图与流程图）保存在私有归档库，公开版不含该文件。

## 5. 复现方式

1. **仿真**：Proteus 打开 `proteus/1.DSN`，加载 `Objects/cx.hex` 对应的固件（或 `firmware/electronic_lock.hex`）运行。
2. **编译**：Keil µVision 打开 `keil/cx.uvproj` → Rebuild → 生成新的 hex。
3. **实物**：按报告中的原理图接线，AT89C51 最小系统 + 4×4 键盘 + 4 位数码管 + 蜂鸣器 + LED，烧录 hex。本项目的验证方式为 **Proteus 仿真 + 实验箱功能验证**（非自行焊接制板）。

## 6. 已知问题（保留原样，未修改代码）

| 位置 | 现象 | 说明 |
|---|---|---|
| `main()` 连续三次错误分支 | `while(1){ Key_Port = 0xff; }` 死循环 | 锁定后无法自行恢复，**必须硬件复位**；这是课程要求的行为，但工程上应改为可复位状态 |
| `#` 回删分支 | `Save_password_table[position--] = 0;` | `position` 为 `char`，在 `position==0` 时自减会下溢为 -1，可能造成显示/状态异常 |
| 万能码 | `9999` 硬编码在校验逻辑里 | 属于调试后门，产品代码不应存在 |
| `Password_table[8]` | 初始化为 `{1,2,3,4,5,6,7,8}`，实际只用前 4 位 | 数组长度与实际使用不一致，易误读 |

## 7. 可以展开讲的技术点（面试/答辩）

1. 为什么把**显示刷新放进 1ms 中断**，而不是主循环？
2. 矩阵键盘**为什么要 10ms 消抖**？不消抖会怎样？行列反转扫描相比逐行扫描的优势？
3. 多个软件计时器共用一个硬件定时器时，**如何避免计数溢出**？（代码里对每个计数器都设了上限保护）
4. 如何用**状态标志位**而不是 `delay()` 实现"错误后 3 秒才能再输入"？
