C**********************************************************************C        
C                                                                      C        
C---------------------- PROGRAM VARIO ---------------------------------C        
C                                                                      C        
C**********************************************************************C        
C
cuh  Program to calculate variograms and residual variograms in
cuh  maximally 8 directions (9, including global variograms)
cuh  and in maximally (nres-1) distance classes
cuh
cuh  By Ute Christina Herzfeld
cuh  
cuh  2000-mar-18, at INSTAAR
cuh
cuh  -------------------------------------------------------------------C
cuh  History section: 
cuh  this is derived from program CROSSVA, by Vera Pawlowsky, and
cuh  modified by Ute Herzfeld:
cuh
C    SEABEAM-POST-PROCESSING VERSION
C    
C    by Ute Christina Herzfeld
c    aboard FS POLARSTERN
c    ANT VI/3
c
c    Alfred-Wegener-Institut fuer Polar- und Meeresforschung
c    Columbusstrasse
c    2850 Bremerhaven 
c    West Germany
c
c    Institut fuer Geologie - Mathematische Geologie -
c    Freie Universitaet
c    Malteserstrasse 74-100
c    1000 BERLIN 46
c 
c                                                                      C        
C   LAST REVISION BY VERA:   FEBRUARY,1986                             C        
C
C   LAST REVISION BY MATTHIAS MIMLER, Aug. 1998 at INSTAAR (Boulder, CO)
C      enabled to print variable number of variogram values
C
C   Last revision by UCH at INSTAAR, to increase number of vario
C      values March 18, 2000
C
cuh: differences to CROSSVA and crossvamod:
cuh  - maximal number of steps is now really variable (needed to delete
cuh    XG(200) and other variables that were needed only in the old 
cuh    line printer graphics
cuh  - new type of input files: only need on output file per direction
cuh  - new type of output file:
cuh    col 1 - lp2, step number
cuh    col 2 - distclass, distance to center of class
cuh    col 3 - m1, (mittelwert)
cuh    col 4 - m2, variogram
cuh    col 5 - m3, residual variogram
cuh    col 6 - dismoy, average distance of pairs used in class
cuh    col 7 - distot, number of pairs used in class
cuh
cuh    notice: in older versions (crossva), we plotted vario vs dismoy



C                                                                      C        
C**********************************************************************C        
C                                                                      C        
C   FUNCTION     COMPUTING OF EXPERIMENTAL VARIOGRAMS FROM SCATTERED   C        
C                S A M P L E S                                         C        
C                AND THEIR  R E S I D U A L S   IN 2-D SPACE.          C        
  
C   PARAMETERS                                                         C        
C                                                                      C        
C   LINE 1:      ICOM    TITLE OF PROJECT        FORMAT (20A4)         C        
C                                                                      C        
C   LINE 2:      ICOM    NAME OF SAMPLING AREA   FORMAT (20A4)         C        
C                                                                      C        
C   LINE 3:      NVAR    NUMBER OF VARIABLES (MAX. 10)                 C        
C                NDIR    NUMBER OF DIRECTIONS (MAX. 9)                 C        
C                        ALLOWED NDIR: 1, 2, 3, 4, 5, 8, 9             C        
C                        NDIR = EVEN, THEN YOU GET N DIFFERENT         C        
C                               DIRECTIONS                             C        
C                        NDIR = ODD, THEN YOU GET THE MEAN VARIOGRAM   C        
C                               AND N-1 VARIOGRAMS IN DIFFERENT        C        
C                               DIRECTIONS                             C        
C                        THE DIRECTIONS ARE AUTOMATICALLY TAKEN IN     C        
C                        A REGULARLY SPACED WAY.                       C        
C                        THE ANGLE OF TOLERANCE IS SET AUTOMATICALLY   C        
C                        SO THAT IT PARTITIONS THE SET OF ALL          C        
C                        POSSIBLE DIRECTIONS.                          C        
C                        REMARK:                                       C        
C                        IT IS ALSO POSSIBLE TO CALCULATE JUST ONE     C        
C                        PARTICULAR DIRECTION. SET NDIR=1 AND USE      C        
C                        PARAMETERS ALPHA AND SPETO ON LINE 5.         C        
C                STEP    RANGE OF CLASSIFICATION FOR DISTANCE          C        
C                        IF YOU PUT STEP=0, THEN STEP IS SET           C        
C                        AUTOMATICALLY TO 1/60 OF THE MAXIMUM          C        
C                        DISTANCE                                      C        
C                        FORMAT ( * )                                  C        
Cm               NRES    number of results/variogram values to be      C
Cm                       calculated                                    C
C                                                                      C        
C   LINE 4:      NAME    NAMES OF VARIABLES                            C        
C                        FOR EACH NAME THERE IS  SPACE FOR 8 LETTERS   C        
C                        FORMAT (10 A8)                                C        
C                                                                      C        
C   LINE 5:              SPECIAL PARAMETERS:                           C        
C                SPETO   SPECIAL ANGLE OF TOLERANCE (< 90)             C        
C                        (IF SPETO=0, IT IS SET AUTOMATICALLY          C        
C                        SO THAT IT PARTITIONS THE SET OF ALL          C        
C                        POSSIBLE DIRECTIONS)                          C        
C                ALPHA   ANGLE FOR ROTATION OF THE COORDINATES         C        
C                CORE    IF CORE IS 1, YOU GET THE CROSS VARIOGRAMS    C        
C                        BETWEEN TWO VARIABLES INSTEAD OF THE SIMPLE   C        
C                        VARIOGRAMS (JUST TRY IT...)                   C        
C                ILOG    LOGARITHMS                                    C        
C                        ILOG=0, NO LOGS ARE CALCULATED                C        
C                        ILOG=J, J>0, LOGARITHM OF VAR.J IS TAKEN      C        
C                        ILOG<0, LOGARITHMS OF ALL VARS. ARE TAKEN     C        
C                        set ilog=1, if want log of variable
c                IACM    ACCUMULATIONS:                                C        
C                        IF IACM>0, THEN THE FIRST TWO VARIABLES ARE   C        
C                        MULTIPLIED AND VARIOGRAMS ARE COMPUTED ON     C        
C                        THIS PRODUCT. THE FIRST VARIABLE IS SUBSTI-   C        
C                        TUTED BY ACCUMULATION.                        C        
C                BORNL   LOWER LIMIT FOR VALUES OF VARIABLES           C        
C                        (IF BORNL = 0, THEN BORNL =-1.E+30)           C        
C                BORNU   UPPER LIMIT FOR VALUES OF VARIABLES           C        
C                        (IF BORNU = 0, THEN BORNU = 1.E+30)           C        
C                                                                      C        
C                        PAR. TO CHARACTERIZE THE GRAPHICAL            C        
C                        REPRESENTATION:                               C        
C                YCHEL   UPPER LIMIT OF THE VARIOGRAM                  C        
C                XCHEL   UPPER LIMIT OF THE DISTANCE                   C        
C                YINF    LOWER LIMIT OF THE LOGARITHMICAL VARIOGRAM    C        
C                                                                      C        
C                K1      COEFFICIENT OF ELLIPTIC OR GEOMETRICAL        C        
C                        ANISOTROPY (IF K1<=0, THEN K1=1)              C        
C                        MULTIPLIES THE Y-COORDINATE AFTER ROTATION    C        
C                        OF COORDINATES (IF ALPHA.NE.0)                C        
C                        FORMAT ( * )                                  C        
C                                                                      C        
C                                                                      C        
C                                                                      C        
Cuh line 6:      infile  name of input file (a50)                      C
cuh following lines only if nec according to no of directions, has
cuh           to be consistent with NDIR above (all a50)
cuh                                        if ndir=8,9  =4,5  =2,3   
c             7:  out1   name of data file vario 0 deg   0      0
c	line  8: out2   name of data file vario 23 deg  45     90
c	line  9: out3   name of data file vario 45 deg  90
c	line 10: out4   name of data file vario 68 deg  135
c	line 11: out5   name of data file vario 90 deg
c	line 12: out6  name of data file vario 113 deg
c	line 13: out7  name of data file vario 135 deg
c	line 14: out8  name of data file vario 158 deg
c       line 15: out9  name of data file vario gobal
c
C-----( END OF PARAMETER FILE - name it 'invario.dat'     )------------C        
C-----( DATA INPUT FILE -                               )--------------C        
c                  X,Y     COORDINATES OF SAMPLES  (F OR E FORMAT)     C        
C                  AN      VARIABLES FOR EACH SAMPLE (MAX. 5,          C        
C                          IF IACM>0)        (F OR E FORMAT)           C        
C                                                                      C        
C----------------------------------------------------------------------C        
C                                                                      C        
C   MISSING VALUES:ALL VALUES <= BORNL OR  >= BORNU ARE CONSIDERED     C        
C                AS M.V. DURING VARIOGRAM CALCULATION (ILOG, IACM:     C        
C                CONSIDER VALUES AFTER TRANSFORMATION FOR BORN)        C        
C                                                                      C         C                                                                      C        
C   CAPACITY:    50000  SAMPLES                                        C        
C                    9  DIRECTIONS                                     C        
C                    3  VARIABLES  
c                 1001  STEPS
C                                                        C        
C                                                                      C        
C   AUFRUF MIT: LGO,<PARAMETERFILE>,<DATENFILE>                        C        
C                                                                      C        
C----------------------------------------------------------------------C        
      REAL PSI, PHI, K1                                                         
      REAL NAME                                                                
      INTEGER DIMT                                                 
      INTEGER EFF, CORE                                                         
cm    NRES eingefuegt
      integer NRES
      CHARACTER*4 FMT(20)                                                       
      character*50 infile,out1,out2,out3,out4,out5,out6,
     1 out7,out8,out9

 
C**********************************************************************C        
C                                                                      C        
C      S P A R E    MEMORY ....                                        C        
C              .... ADJUST   D I M E N S I O N   LINES                 C        
C                                                                      C        
C**********************************************************************C        
C                                                                      C        
C** FIRST PARAMETER:  NUMBER OF SAMPLES                                C        
C** SECOND PARAMETER: NUMBER OF VARIABLES                              C        
C                                                                      C        
           dimension an(50000,3)         
C                                                                      C        
C** PARAMETER: NUMBER OF SAMPLES                                       C        
C                                                                      C        
           real x(50000), y(50000)                    
C                                                                      C        
cuh parameter nres: number of steps (use one more, 1st is lost)
cuh 
C** FIRST PARAMETER: must be NRES*9 (here: maximal nres=1001)
cuh                                           
C** SECOND PARAMETER: NUMBER OF VARIABLES                              C        
C                                                                      C        
       dimension distot(9009,3),eff(9009,3),s1(9009,3),s2(9009,3) 
C                                                                      C        
C                                                                      C        
C****   ...AND ADJUST FIRST DIMENSION LINE OF SUBROUTINE VARIO...      C        
C                                     AND  OF SUBROUTINE COREG...      C        
C                                                                      C        
C****     AND ADJUST IN MAIN PROGRAM THE LINE: DATA DIMT...            C        
C                                                                      C        
C....NOW ALL SHOULD BE READY: GOOD LUCK.                               C        
C                                                                      C        
C**********************************************************************C        
C                                                                               
      DIMENSION ICOM(40), NAME(10)                                              
      COMMON /DATA/ PSI (10), PHI (10)                                          
      COMMON /HEAD/ ICOM                                                        
      COMMON /PARA/ STEP,BORNL,BORNU,NAME,IACM,ILOG,IDIR,XCHEL,YCHEL,           
     &         YINF, K1, ALPHA,ROTWI                                     
      DATA DIMT /50000/                                                          
      DATA ICOM /40* 4H    /,FMT /20*'    '/                                    
    2 FORMAT (20A4)                                                             
    1 FORMAT (1H1,2(/,1H0,20A4),                                                
     &/,1H0,'VARIABLES: ',5(2X,A8),/,12X,5(2X,A8))                             
    5 FORMAT(1H0,' LOG ACM DIR      STEP      LOWER BORN',                      
     &'      UPPER BORN        K1     ALPHA',/1H ,                              
     &3I4,F10.3,2E16.3,2F10.3,/,1H )                                            
    6 FORMAT (1H1)                                                              
    7 FORMAT (1H0,19HNUMBER OF SAMPLES :,I8,/                                   
     &      36H0SAMPLES WITH EQUAL COORDINATES ARE:,/ 1H )                      
    8 FORMAT (1H ,18HANGLE     DEGREE: ,10F10.1)                                
   10 FORMAT (1H ,18HDIRECTION DEGREE: ,10F10.1)                                
   11 FORMAT (1H ,A4,3E10.3)                                                    
   12 FORMAT (A3)                                                               
 1002 FORMAT (10 A8)                                                            
C                                                                               
C**** INDATA:  INPUT CHANNEL FOR DATA             ****                 C        
C                                                                               
c      INDATA = 9    
C
	open(unit=27,file='invario.dat',status='old') 

c                 PARAMETERS READ                                              
C                                                                               
      READ (27,2)ICOM  
      READ(27,*) NVAR,NDIR,STEP,NRES
      READ (27,1002)  (NAME (I),I = 1,NVAR)
      READ (27,*)SPETO,ALPHA,CORE,ILOG,IACM,BORNL,BORNU,YCHEL,              
     & XCHEL,YINF,K1                                                            
      ROTWI=ALPHA                                                               
      IF (NDIR.LE.0)   NDIR = 1                                                 
      IDIR = NDIR                                                               
      CALL SETDI ( NDIR,SPETO,PHI,PSI )                                         
      IF (BORNL.EQ.0.)   BORNL =-1.E+30                                         
      IF (BORNU.EQ.0.)   BORNU = 1.E+30                                         
      IF (K1.LE.0.)   K1 = 1                                                    
cuh
      read (27,'(a)') infile	
      if (idir.eq.1) read (27,'(a)') out1
      if (idir.eq.2) read (27,'(a)') out2
      if (idir.eq.3) read (27,'(a)') out3
      if (idir.eq.4) read (27,'(a)') out4
      if (idir.eq.5) read (27,'(a)') out5
      if (idir.eq.8) read (27,'(a)') out6
      if (idir.eq.8) read (27,'(a)') out7
      if (idir.eq.8) read (27,'(a)') out8
      if (idir.eq.9) read (27,'(a)') out9

                          
      if (idir.eq.1) open (unit=41,file=out1,status='new')
      if (idir.eq.2) open (unit=42,file=out2,status='new')
      if (idir.eq.3) open (unit=43,file=out3,status='new')
      if (idir.eq.4) open (unit=44,file=out4,status='new')
      if (idir.eq.5) open (unit=45,file=out5,status='new')
      if (idir.eq.8) open (unit=46,file=out6,status='new')
      if (idir.eq.8) open (unit=47,file=out7,status='new')
      if (idir.eq.8) open (unit=48,file=out8,status='new')
      if (idir.eq.9) open (unit=49,file=out9,status='new')


cuh      PRINT 5, ILOG,IACM,IDIR,STEP,BORNL,BORNU,K1,ALPHA  
                       
cuh      Print 10, ((PHI(I)+ROTWI),I=1,IDIR)
                                       
cuh      PRINT  8, (PSI(I),I = 1,IDIR)
                                        
C                                                                               
C                  DATA CONCERNING SAMPLES ARE READ                             
C                  (COORDINATES,GRADE AND/OR LENGTH)                            
C                                                                               
cuhcuh
      open(28,file=infile,status='old')      

      ALPHA = 3.1415926535 * ALPHA / 180.                                       
      DO 104 N = 1,DIMT                                                         
     
                 read(28,*,end=124) x(n),y(n),an(n,1)
                     
       GOTO 103                                                                 
                                                           
                              
C                                                                               
C       LOGARITHMIC TRANSFORMATIONS                                             
C                                                                               
  103    IF( ILOG.EQ.0 ) GOTO 104                                               
          IF( ILOG.LT.0 ) GOTO 1103                                             
                       
           if(an(n,ilog).le.bornl.or.an(n,ilog).ge.bornu) goto 104
                    an(n,ilog) = alog (an (n,ilog))     
          GOTO 104                                                              
 1103       DO 1104 J=1,NVAR                                            
 1104         AN (N,J) = ALOG ( AN (N,J) )                              
  104    CONTINUE                                                               
C                                                                               
C                                                                               
cuh       PRINT 110, DIMT                                                     
         
  110 FORMAT(1H ,10H EXACTLY   ,I5,24H DATA ????? OR MORE ??    ,/,             
     & 55H    IF MORE: ADJUST DIMENSION AND DATA DIMT LINES !!!   )             
  124 N = N - 1                                                                 


cuh      PRINT 7,N                                                                 

      IF ( CORE.GE.1 )  GOTO 3000                                               

      call vario (x,y,an,n,nvar,distot,eff,s1,s2,NRES)


      GOTO 3010                                                                 
                       
 3000 continue  
                                  
 3010 CONTINUE                                                                  
      STOP                                                                      
      END                                                                       
C                                                                               
C----------------------------------------------------------------------C        
C                                                                               
      SUBROUTINE VARIO (X,Y,T,N,NV,DISTOT,EFF,S1,S2,NRES)                       
                       
C                                                                               
C----------------------------------------------------------------------C        
C                                                                               
C   FUNCTION      COMPUTING OF VARIOGRAMS IN SEVERAL DIRECTIONS                 
C                                                                               
C   PARAMETERS    X  Y     COORDINATES OF SAMPLES                               
C                 T        VARIABLE                                             
C                 N        NUMBER OF SAMPLES                                    
C                 NV       NUMBER OF VARIABLES                                  
C                 MOY      MEAN OF T-VALUES FOR THE SET OF SAMPLES              
C                 VRNCE    CORRESPONDING VARIANCE                               
C                                                                               
C             FOR EACH DIRECTION                                                
C             FOR EACH INTERVAL OF DISTANCE,                                    
C             FOR EACH VARIABLE,                                                
C                 EFF      NUMBER OF PAIRS OF SAMPLES                           
C                 DISMOY   AVERAGE DISTANCE                                     
C                 M1       DRIFT                                                
C                                                                               
C   CAPACITY       N  SAMPLES                                                   
C                 10  VARIABLES                                                 
C                  9  DIRECTIONS                                                
C                                                                               
C     RECQ. ROUTINES         GRAPH                                              
C                                                                               
C----------------------------------------------------------------------C        
      INTEGER EFF,CLASS    
      real distclass                                                                                                     
      CHARACTER XTIT(30),VATIT(30)                                              
          
      REAL NAME                                                                 
      REAL MOY,M1,M2,M3,K1                                                      
      REAL PSI,PHI                                                              
C**********************************************************************C        
C                                                                               
C     A D J U S T  (ONLY) FIRST DIMENSION LINE.                                 
C                                                                               
C** FIRST PARAMETER: NUMBER OF SAMPLES                                          
C** SECOND PARAMETER: HAS TO BE NV                                              
C                                                                               
      DIMENSION  T(50000,NV)                                                     
C                                                                               
C**********************************************************************C        
C
cuh   ADJUST first dimension parameter in distot, eff,s1,s2
cuh   this must be NRES*9, for maximal nres=1001, this is 9009

                                                                             
      DIMENSION DISTOT(9009,NV),EFF(9009,NV),S1(9009,NV),S2(9009,NV)                
      DIMENSION T1(9), CA(9), SA (9)                                            
      DIMENSION ICOM(40),NAME(10)                                               
      DIMENSION X(N),Y(N)                                               
      DIMENSION TMIN(10),TMAX(10),SM(10),SIGMA(10),IP(10)                       
      DIMENSION MOY (10),VRNCE (10)                                             
                         
      COMMON /DATA/ PSI (10),PHI (10)                                           
      COMMON /HEAD/ ICOM                                                        
      COMMON /PARA/ STEP,BORNL,BORNU,NAME,IACM,ILOG,IDIR,XCHEL,YCHEL,           
     &         YINF,K1,ALPHA,ROTWI                                        
 
      DATA XTIT /'A','V','E','R','A','G','E',                                   
     &' ','D','I','S','T','A','N','C','E',5*' ',                                
     &'&','=','R','E','S',4*' '/                                                
      DATA VATIT /4*' ','V','A','R','I','O','G','R','A','M',17*' '/             
    3 FORMAT (13X,'DISTANCE CLASS        NB. OF PAIRS      DRIFT',              
     &'        VARIOGRAM   VARIOGRAM (RES)   AVERAGE DISTANCE',//)              
    4 FORMAT (7X,F9.2,5H ----,F9.2,4X,I8,7X,E10.3,4X,                           
     &        E13.4,3X,E13.4,3X,F10.3)                                          
  500 FORMAT (1H1,57X,17HV A R I O G R A M  //,                                 
     &        28X,20A4,//28X,20A4//,                                            
     &        41X,19H( WITH A FIELD OF  ,F5.1,                                  
     &        31H DEGREES IN EACH DIRECTION ),                                  
     &       /101X,16(1H.))                                                     
  501 FORMAT (101X,1H.,14X,1H.)                                                 
  502 FORMAT (101X,1H.,3X,A8,3X,1H.)                                            
  499 FORMAT (1H ,27HSTEP                    =  ,E10.4,                         
     &  7X,19HNUMBER OF VALUES =  ,I5,32X,16(1H.)/)                             
  503 FORMAT (1H ,27HLOWER LIMIT FOR T       =  ,E10.4,                         
     &    7X,7HXMIN  =,E12.4,7X,7HXMAX  =,E12.4/)                               
  504 FORMAT (1H ,27HUPPER LIMIT FOR T       =  ,E10.4,                         
     &    7X,7HYMIN  =,E12.4,7X,7HYMAX  =,E12.4,11X,14(1H.))                    
  505 FORMAT (1H ,100X,1H.,12X,1H.)                                             
  506 FORMAT (1H ,27HGENERAL MEAN OF T       =  ,E10.4,                         
     &    7X,7HTMIN  =,E12.4,7X,7HTMAX  =,E12.4,11X,                            
     &        1H.,3X,F5.1,4X,1H.)                                               
  507 FORMAT (1H ,27HGENERAL VARIANCE OF T   =  ,E10.4,                         
     &    63X,14(1H.)/)                                                         
      CALPHA = COS (ALPHA)                                                      
      SALPHA = SIN (ALPHA)                                                      


c#########################################################################
C                                                                               
      KI = 1                                                                    
      KIN = NRES                                                               
      DO 101 LP1 = 1,IDIR                                                       
         APSI = 3.1415926535 * PSI (LP1) / 180.                                 
         T1 (LP1) = COS (APSI) - 1.E-10                                         
         APHI = 3.1415926535 * PHI (LP1) / 180.                                 
         CA (LP1) = COS (APHI)                                                  
         SA (LP1) = SIN (APHI)                                                  
C                                                                               
         DO 100 LP2 = KI,KIN                                                    
            DO 100 J= 1,NV                                                      
            EFF (LP2,J) = 0                                                     
            DISTOT (LP2,J) = 0.                                                 
            S1 (LP2,J) = 0.                                                     
  100       S2 (LP2,J) = 0.                                                     
      KI = KI + NRES                                                            
      KIN = KIN + NRES                                                        
  101 CONTINUE                                                                  
C                                                                               
c##########################################################################

C                 LP1 = LOOP ONE                                                
C                 IP = NUMBER OF VALUES OF A VARIABLE                           
C                 SM = SUM OF VALUES                                            
C                 SIGMA = SUM OF SQUARES                                        
C                                                                               
         XMIN = 3.*1.E30                                                        
         YMIN = 3.*1.E30                                                        
         XMAX = -3.*1.E30                                                       
         YMAX = -3.*1.E30                                                       
         DO  1100 J= 1,NV                                                       
         TMIN (J) = 3.*1.E30                                                    
         TMAX (J) = -3.* 1.E30                                                  
         IP (J) = 0                                                             
         SM (J) = 0.0                                                           
 1100    SIGMA (J) = 0.0                                                        
         DO 1112  JJ = 1,N                                                      
                  XMIN = AMIN1 (XMIN,X (JJ))                                    
                  YMIN = AMIN1 (YMIN,Y (JJ))                                    
                  XMAX = AMAX1 (XMAX,X (JJ))                                    
 1112             YMAX = AMAX1 (YMAX,Y (JJ))                                    
      IF (STEP.GT.0.0) GOTO 112                                                 
      STEP = SQRT ((XMAX - XMIN)**2 + (YMAX - YMIN)**2) / 60.                   
C                                                                               
C                 ALL POSSIBLE  PAIRS OF SAMPLES ARE INVESTIGATED               
C                 WITHIN THE LOOPS LP1 AND LP2.  IF THE                         
C                 DIRECTION OF THE LINE JOINING THE POINTS OF                   
C                 A PAIR FALLS BETWEEN THE ANGULAR LIMITS                       
C                 AROUND PHI,THE PAIR IS RETAINED AND ITS                       
C                 CONTRIBUTION TO VARIOGRAM AND DRIFT IS CALCULATED             
C                 FOR EVERY VARIABLE                                            
C                                                                               
  112    DO 145 LP1 = 1,N                                                       
                   DO 1110 LP2= 1,NV                                            
                   IF ( T (LP1,LP2).GE.BORNU .OR.                               
     &                  T (LP1,LP2).LE.BORNL  )       GOTO 1110                 
                   TMIN (LP2) = AMIN1( TMIN (LP2),T (LP1,LP2) )                 
                   TMAX (LP2) = AMAX1( TMAX (LP2),T (LP1,LP2) )                 
                   IP (LP2) = IP (LP2) + 1                                      
                   SM (LP2) = SM (LP2) + T (LP1,LP2)                            
                   SIGMA (LP2) = SIGMA (LP2) + T (LP1,LP2) * T (LP1,LP2)        
 1110              CONTINUE                                                     
            I2 = LP1 + 1                                                        
            IF (I2.GT.N) GO                                   TO 145            
               DO 144 LP2 = I2,N                                                
                     DXT = X (LP1) - X (LP2)                                    
                     DYT = Y (LP1) - Y (LP2)                                    
                 DXT2 = DXT * CALPHA + DYT * SALPHA                             
                 DYT2 = K1 * (DYT * CALPHA - DXT * SALPHA)                      
                     D2 = DXT2 * DXT2 + DYT2 * DYT2                             

           if (d2.lt.1.e-20) goto 144       
                 D1 = SQRT (D2)                                          
C                                                                               
C   DIRECTIONS LOOP                                                             
C                                                                               
           DO 1145 LDIR = 1,IDIR                                                
C    IC IS NUMBER OF DIRECTION CLASS                                            
                        IC = INT (D1 / STEP + 0.5) + 1                          
                        IF (IC.GT.NRES) GO                TO  144            
C    CC IS COSINE OF DIFFERENCE BETWEEN TWO ANGLES                              
        CC = DXT2 * CA (LDIR) / D1 + DYT2 * SA (LDIR) / D1                      
                        IF (ABS (CC).LT.T1 (LDIR))  GO       TO 1145            
C    IC: VALUES FOR ALL DIRECTIONS ARE PUT IN THE SAME COLUMN                   
                IC = IC + NRES * (LDIR-1)                                      
C                                                                               
C    VARIABLES LOOP                                                             
C                                                                               
                     DO 1144  LP3= 1,NV                                         
      IF (( T (LP1,LP3).GE.BORNU.OR.T (LP1,LP3).LE.BORNL )     .OR.             
     &    ( T (LP2,LP3).GE.BORNU.OR.T (LP2,LP3).LE.BORNL )) GOTO 1144           
           DELTZ  = ( T (LP1,LP3) -T (LP2,LP3) ) * SIGN(1.,CC)                  
C     EFFECTIVE NUMBER OF PAIRS                                                 
           EFF (IC,LP3) = EFF (IC,LP3) + 1                                      
C     SUM OF DISTANCES                                                          
           DISTOT (IC,LP3) = DISTOT (IC,LP3) + D1                               
C     SUM OF DRIFT RESIDUALS                                                    
           S1 (IC,LP3) = S1 (IC,LP3) + DELTZ                                    
C     SUM OF SQUARES (VARIOGRAM)                                                
           S2 (IC,LP3) = S2 (IC,LP3) + DELTZ  * DELTZ                           
 1144               CONTINUE                                                    
 1145         CONTINUE                                                          
  144       CONTINUE                                                            
  145    CONTINUE                                                               
C         MEAN AND VARIANCE ARE CALCULATED FOR EVERY VARIABLE                   
            DO 1111 J= 1,NV                                                     
         IF (IP(J).GT.0) GOTO 111                                               
           MOY (J) = 0.0                                                        
           VRNCE (J) = 0.0                                                      
           GOTO  1111                                                           
  111       MOY (J) = SM (J) / FLOAT( IP (J) )                                  
            VRNCE (J) = ( 1./ (IP (J) * (IP (J)- 1) )) *                        
     &                        ( IP (J) * SIGMA (J) - SM (J) * SM (J) )          
 1111       CONTINUE                                                            
C                                                                               
C                 PRINTING THE RESULTS                                          
C                 
      	
	                                                             
                DO 155 J= 1,NV                                                  
                KI = 1                                                          
                KIN =NRES                                                      
                     DO 1147  LP1 = 1,IDIR                                      
cuhuhuh  
cuh          PRINT 500,ICOM,PSI (LP1)                                  
                                           
cuh          PRINT 501                                                        
          
cuh          PRINT 502,NAME (J)                                               
cuh         PRINT 501                                                        
cuh         PRINT 499,STEP,IP(J)                                             
cuh           PRINT 503,BORNL,XMIN ,XMAX                                           
cuh           PRINT 504,BORNU,YMIN ,YMAX 
cuh           PRINT 505                                                        
cuh          PRINT 506,MOY (J),TMIN(J),TMAX(J),(PHI(LP1)+ROTWI)               
cuh           PRINT 505                                                        
cuh            PRINT 507,VRNCE (J)                                              
cuh          PRINT 3                                                                                                                                                          
         VARMAX = -1.E+30                                                       
         VARMIN = +1.E+30                                                       
         CLASS = 0                                                              
         DO 153 LP2 = KI,KIN                                                    

cuh
	      distclass=step*real(class)   
              CLASS = CLASS + 1   
                                          
            IF (EFF (LP2,J).LT.2) GO                          TO 149
                  
               M1 = S1 (LP2,J) / FLOAT (EFF (LP2,J))                            
               M2 = 0.5 * S2 (LP2,J) / FLOAT (EFF (LP2,J))                      
               M3 = 0.5 * (S2 (LP2,J) - S1 (LP2,J) * S1 (LP2,J)                 
     &              / FLOAT (EFF (LP2,J)) )  / FLOAT (EFF (LP2,J))              
               DISMOY = DISTOT (LP2,J) / FLOAT (EFF (LP2,J))                      
               VARMAX = AMAX1 (VARMAX,M2,M3)                                    
               VARMIN = AMIN1 (VARMIN,M2,M3)                                                                          
cuhuhuh   
         	 kkv= lp1 +40
        	 kkr= lp1 +60

cuh            do not print results for the first "half" distance class 

               if (distclass.lt.(0.9*step)) goto 153

cuh               PRINT 4,BINF,BSUB,EFF (LP2,J),M1,M2,M3,DISMOY              
	write (kkv,'(i5,5f20.6,i7)') lp2,distclass,m1,m2,m3,dismoy,
     &          eff(lp2,j)
	write (kkr,'(3f20.6,i7)') distclass,m3,dismoy,eff(lp2,j)

  149       continue                          
  153       CONTINUE   


 1146    KI = KI + NRES                                                      
      KIN = KIN + NRES                                                        
 1147 CONTINUE                                                                  
  155    CONTINUE                                                                                               
      RETURN                                                                    
      END                                                                       
cuh-----------------------------------------------------------------------
   
C----------------------------------------------------------------------C        
C                                                                               
      SUBROUTINE LISTDO (X,Y,L1,L2 )                                       
C                                                                               
C----------------------------------------------------------------------C        
C                                                                               
      DIMENSION X(L2),Y(L2)                                            
C                                                                               
      write (5,*) X(L1),Y(L1),X(L2),Y(L2)        
  10  format (2f12.3,/,2f12.3) 
      RETURN                                                                
      END                                                                       
C                                                                               
C----------------------------------------------------------------------C        
C                                                                               
      SUBROUTINE SETDI ( NDIR,SPEPSI,PHI,PSI )                                  
C                                                                               
C----------------------------------------------------------------------C        
C                                                                               
C    SUBROUTINE THAT SETS DIRECTIONS AND ANGLES OF TOLERANCE.                   
C                                                                               
C----------------------------------------------------------------------C        
C                                                                               
      DIMENSION PHI (10),PSI (10)                                               
      PHI (NDIR) = 0.0                                                          
      PSI (NDIR) = 90.                                                          
      IF ( NDIR.NE.1 ) GOTO 1                                                   
      GOTO 15                                                                   
    1 IF (( NDIR.LE.5 ).OR.                                                     
     &    ( NDIR.GE.8.AND.NDIR.LE.9 )) GOTO 2                                   
      STOP '*** STOP: WRONG NUMBER OF DIRECTIONS ***'                           
    2 RDIR = FLOAT (NDIR)                                                       
      REST = RDIR / 2.                                                          
      REST = REST - FLOAT ( IFIX(REST) )                                        
      IF ( REST.EQ.0.) GOTO 3                                                   
      NDIR = NDIR - 1                                                           
      RDIR = RDIR - 1.                                                          
    3 DPHI = 180./ RDIR                                                         
      DPSI = 90./ RDIR                                                          
      SPHI = 0.                                                                 
          DO 10 I= 1,NDIR                                                       
          PHI (I) = SPHI                                                        
          PSI (I) = DPSI                                                        
          SPHI = SPHI + DPHI                                                    
   10 CONTINUE                                                                  
   15 IF ( SPEPSI.LE.0.) GOTO 4                                                 
          DO 20 I= 1,NDIR                                                       
   20     PSI (I) = SPEPSI                                                      
    4 RETURN                                                                    
      END                                                                       
C                                                                               
            
                     
           
