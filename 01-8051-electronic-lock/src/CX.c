#include<reg51.h>  	   //包含头文件
//宏定义uchar和uint
#define uchar unsigned char
#define uint unsigned int
//宏定义ON和OFF
#define	ON	0
#define	OFF	1

#define Key_Port   P1  	  //定义键盘的接口
sbit LED0 =	  P3^0;		  //定义LED指示灯接口
sbit BUZZ = P3^1;		  //定义蜂鸣器接口

//数码管的位选
sbit seg1 = P2^0;		 
sbit seg2 = P2^1;		  
sbit seg3 = P2^2;
sbit seg4 = P2^3;		  

uchar Display_table[4]={0,0,0,0};//存储要显示的数值
uchar Save_password_table[8]={0};//存储要显示的数值
uchar Password_table[8]={1,2,3,4,5,6,7,8};//{1,2,3,4,5,6,7,8};//存储密码
uint ms_delay;
char position  = 0;                             //要显示的位置
uint LED_on_delay=0;                            //密码正确后LED亮一秒延时
uchar BUZZ_on_num=0;                            //蜂鸣器响的次数
uint BUZZ_delay; 	                       		 //蜂鸣器延时
bit  password_wrong_flag=0;                     //密码错误标志
uint password_wrong_delay=0;                    //错误延时时间
uint Enter_password_outtime_delay=0;            //密码输入超时
uint Close_Lock_time=0;//


const uchar SEG_Table[]={0x3f,0x06,0x5b,0x4f,0x66,0x6d,0x7d,0x07,0x7f,0x6f,0x77,0x7c,0x39,0x5e,0x79,0x71,0x00};//
//0     1    2	  3		4	5	6		7	8	9	A	 B		C	D	E	  F	不显示 r	-
const	uchar keynum_tab[16]={0xd7,0xee,0xde,0xbe,0xed,0xdd,0xbd,0xeb,0xdb,0xbb,0x7e,0x7d,0x7b,0x77,0xb7,0xe7};//表格数据是根据按键值和个按键的定义的出来的，不支持组合键
// 0	1     2	  3     4    5    6    7    8    9    A    B    C    D    #    *     

//************************************
//定时器初始化
//************************************
void Time0_init()
{
  TMOD |= 0x01;			 //使用模式1,16位定时器
  TH0   = (65536-1000)/256;	//定时器装入初始值1ms
  TL0   = (65536-1000)%256;
  EA    = 1;  //总中断
  ET0   = 1;//定时器0初始化
  TR0   = 1;
  
}
/**************************************************
函数名称：delay(uint x)
功能：简单延时函数
***************************************************/
void  delay(uint x)
{
  uint y,z;
  for (y=x; y>0; y--)
    for (z=110; z>0; z--);
}
/**************************************************
函数名称：uchar Key_scan()
功能：矩阵按键扫描, 按键按下则返回按键值不按就返回0xff
***************************************************/
uchar Key_scan()
{
  static uchar key_down;
  uchar key_num=0xff,tem,i;   
  
  Key_Port = 0xf0;
  if(Key_Port != 0xf0)//说明有按键按下
  {						
    if(key_down==0)	//有按键按下
    {
      delay(10); 	//延时按键消抖
      tem = Key_Port;
      Key_Port = 0xff;
      Key_Port = (tem|0x0f);
      key_num = Key_Port;
      key_down=1;	 
      
      for(i=0;i<16;i++)
      {
        if(keynum_tab[i]==key_num)
          break;		
      }
      return i;
      
    }
    else
      return 0xff;		  		   
  }	
  else
  {
    key_down = 0;
    return 0xff;		
  }  
  
}
/**************************************************
函数名称：void Dis_data(void)
功能：数码管动态扫描
***************************************************/
void Dis_data(void)
{
  static uchar temp=0;
  P0=0X00;       //消隐
  switch(temp)
  {
  case 0:seg1=0;seg2=1;seg3=1;seg4=1;P0=SEG_Table[Display_table[0]];break;
  case 1:seg1=1;seg2=0;seg3=1;seg4=1;P0=SEG_Table[Display_table[1]];break;
  case 2:seg1=1;seg2=1;seg3=0;seg4=1;P0=SEG_Table[Display_table[2]];break;
  case 3:seg1=1;seg2=1;seg3=1;seg4=0;P0=SEG_Table[Display_table[3]];break;
  
  }
  temp++; 
  if(temp>3)temp=0;	//扫描4个数码管
}
/**************************************************
函数名称：void Buzz_on(void)
功能： 控制蜂鸣器发声
***************************************************/
void Buzz_on(void)
{
  static uchar temp_step=0;
  static uchar temp_on_num=0;
  switch(temp_step)
  {
  case 0:
    temp_on_num=BUZZ_on_num*2;
    if(temp_on_num)
    {
      //BUZZ=ON;
      BUZZ_delay=0;
      temp_step++;
    }
    break;
  case 1:
    if(temp_on_num&0x01)//奇数
    {
      BUZZ=OFF;
    }
    else
    {
      BUZZ=ON;
    }
    if(!temp_on_num)
    {
      temp_step=0;
      BUZZ=OFF;
      BUZZ_on_num=0;
    }
    else
    {
      if(BUZZ_delay>200)
      {
        BUZZ_delay=0;
        temp_on_num--;
      }	
    }
    break;
  } 	
}


/**************************************************
函数名称：	void main(void)
功能：主函数，实现主要的程序控制流程
***************************************************/						  
void main(void)
{
  uchar key_num,i,Input_num=0;
  uchar	Right_password=0; 
  uchar LED_Flashnum=0;
  uchar unLock = 0;	  //开锁标志
  LED0 = 1;		//关闭指示灯
  BUZZ = 1;		//关闭蜂鸣器
  Time0_init();	//初始化定时器
  while(1)
  {
    
    key_num = Key_scan(); 		//获取按键值	
    if( key_num <10	) 	//显示键值
    {	
      LED_Flashnum=0;		      
      Save_password_table[position++] = key_num; //显示按键值
      if(position>4) position=4;
      else
      {
        Display_table[0]=Display_table[1];
        Display_table[1]=Display_table[2];
        Display_table[2]=Display_table[3];
        Display_table[3] =8; //显示按键值
      }
    }
    else if(key_num==11)//按键B	 万能开锁
    {
      unLock = 1;	//置位开锁
      Right_password=0;
      LED0=ON;
      position=0;
      
    }
    else if(key_num==12)//按键C 复位
    {
      unLock = 0;
      Right_password=0;
      LED0=OFF;  		//关闭LED
      position=0;
      for(i=0;i<4;i++)
      {
        Display_table[i] = 0;
        Save_password_table[i] = 0;
      }
    }
    else if(key_num==13)//按键D	  确定
    {
      if(unLock)   //如果已经开锁就允许设定密码
      {
        for(i=0;i<4;i++)
        {
          Password_table[i] = Save_password_table[i];//修改密码
          Display_table[i] = 0;
          Save_password_table[i] = 0;
        }
        Right_password=0;
        position=0;
        unLock = 0;	//清除
        LED0=OFF;	//关闭显示
      }
      else
      {
        for(i=0;i<4;i++)
        {
          if(Save_password_table[i]!=Password_table[i])
            break;
        }
        if(i<4)//判断万能钥匙
        {
          for(i=0;i<4;i++)
          {
            if(Save_password_table[i]!=9)
              break;
          }	
        }
        if(i<4)	   //说明输入的密码错误Error
        {  
          Display_table[0]=15;
          Display_table[1]=15;
          Display_table[2]=15;
          Display_table[3]=15;									
          
          Right_password=0;
          position=0;
          LED0=OFF;	
          LED_on_delay=0;//
          LED_Flashnum=6;//闪烁次数 LED_Flashnum/2
          BUZZ_on_num=1;//蜂鸣器响1次
          password_wrong_flag=1;
		  if(++Input_num>=3)  //密码输入错误就报警
		  {
				
		  	Input_num = 0;
				LED0=ON;
			  BUZZ=0;//蜂鸣器响6次
				while(1)
				{
					  Key_Port = 0xff;
					}
		  }
        }
        else		 //密码输入正确
        {
          for(i=0;i<4;i++)
          {			
            Display_table[i]=0; //'0'
            Save_password_table[i]=0;
          }
		  Input_num = 0;
          Right_password=1;
          LED0=ON;
          LED_on_delay=0;		
          position=0;
        }
      }
    }
    else if(key_num==10)//#	  清零,回删
    {
      Display_table[3]=Display_table[2];
      Display_table[2]=Display_table[1];
      Display_table[1]=Display_table[0];
      Display_table[0]=0;
      Save_password_table[position--]=0;
      
    }
    
    if(position)//position!=0说明有数字按键按下此时开始密码超时计算，有按键就清零
    {
      if(Enter_password_outtime_delay>9999)//10000ms=10s	后超时报警
      {
        for(i=0;i<position;i++)
        {
          Save_password_table[i]=0;
        }
        position=0;
        Enter_password_outtime_delay=0;
        BUZZ_on_num=4;//蜂鸣器响4声报警
      }
    }
    else
    {
      Enter_password_outtime_delay=0;
    }
	if(password_wrong_flag)	 	//密码错误三秒后才允许输入
    {
      if(password_wrong_delay>2999)
      {
        password_wrong_delay=0;
        password_wrong_flag=0;
      }
      else
      {
        if(key_num!=0xff) //说明有按键按下重新禁止输入，重新计时3s
        {
          password_wrong_delay=0;	   //有按键就清除计时
          Close_Lock_time=0;
        }	
      }
    }
    else
    {
      
      password_wrong_delay=0;
      if(key_num!=0xff)  //有按键按下，就清零计数
      {
        Enter_password_outtime_delay = 0;	
      }		   	
    }	
    if(ms_delay>500)  //延时一段时间
    {	
      ms_delay = 0;
      if(LED_Flashnum>0)
      {
        if(LED_Flashnum&0X01)//如果是奇数
        {
          for(i=0;i<4;i++)			
            Display_table[i]=16;  //关闭显示 
        }
        else
        {
          //显示“ERROR”
          Display_table[0]=15;
          Display_table[1]=15;
          Display_table[2]=15;
          Display_table[3]=15;
          
        }
        LED_Flashnum--;
        if(LED_Flashnum==0)
        {
          for(i=0;i<8;i++)			
            Display_table[i]=0;   
        }
      }

      if(Right_password) //如果密码正确
      {
        if(LED_on_delay>999) 	//大于1s关灯
        {
          LED0=OFF;	
        }	
      }
	  //万能开锁30s后关锁
      if(Close_Lock_time>300)	
      {
        Close_Lock_time=0;
        Right_password=0;
        position=0;	
        LED0=OFF;
		unLock = 0;	//清除
      }
      Buzz_on();//驱动蜂鸣器
    }  
  }
  
}


/**************************************************
函数名称：void time0(void)interrupt 1
功能：1ms定时器中断
***************************************************/
void time0(void)interrupt 1
{
  static uchar temp_tiem_counter=0;
  if(++temp_tiem_counter>=100) //100ms
  {
    temp_tiem_counter=0;
    Close_Lock_time++;
  }
  
  if(++ms_delay>1000)	//防止溢出
  {
    ms_delay=0;
  }
  if(++LED_on_delay>10000)
  {
    LED_on_delay=10000;
  }
  if(++BUZZ_delay>9999)//蜂鸣器延时
  {
    BUZZ_delay=10000;
  }
  if(++password_wrong_delay>9999)
  {
    password_wrong_delay=10000;
  }
  if(++Enter_password_outtime_delay>20000)
  {
    Enter_password_outtime_delay=200000;
  }
  Dis_data();	//显示函数
  //蜂鸣器装入初始数值
  TH0=(65536-1000)/256;
  TL0=(65536-1000)%256;
}
