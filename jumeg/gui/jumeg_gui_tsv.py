#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
which graphiccard to select
https://askubuntu.com/questions/1038271/intel-amd-hybrid-graphics-ubuntu-18-04
xrandr --listproviders
xrandr --setprovideroffloadsink 1 0

DRI_PRIME=0 glxinfo | grep "OpenGL renderer"
DRI_PRIME=1 glxinfo | grep "OpenGL renderer"

DRI_PRIME=0 glmark2 --fullscreen
DRI_PRIME=1 glmark2 --fullscreen
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 11.04.19
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------


import os,sys,argparse

#import OpenGL
#OpenGL.ERROR_CHECKING = False
#OpenGL.ERROR_LOGGING = False
#--- OGL debug / logging SLOW !!!
#OpenGL.FULL_LOGGING = True #False  # True

#OpenGL.ERROR_ON_COPY = True

import wx
from copy import deepcopy

from pubsub import pub

import wx.lib.agw.pybusyinfo as PBI

#--- jumeg cls

#--- jumeg wx stuff
from jumeg.gui.wxlib.jumeg_gui_wxlib_main_frame      import JuMEG_wxMainFrame
from jumeg.gui.wxlib.jumeg_gui_wxlib_main_panel      import JuMEG_wxMainPanel

from jumeg.gui.tsv.utils.jumeg_tsv_utils_io_data         import JuMEG_TSV_Utils_IO_Data
from jumeg.gui.tsv.wxlib.jumeg_tsv_wxlib_plot2d          import JuMEG_TSV_wxPlot2D
from jumeg.gui.tsv.wxutils.jumeg_tsv_wx_utils            import jumeg_tsv_wxutils_openfile,jumeg_tsv_wxutils_dlg_plot_settings
from jumeg.gui.tsv.wxutils.jumeg_tsv_wxutils_dlg_options import GroupDLG # ChannelDLG

import logging
from jumeg.base import jumeg_logger
logger = logging.getLogger('jumeg')


__version__="2019-09-13-001"

#class JuMEG_wxTSVPanel(JuMEG_wxMainPanel):
class JuMEG_wxTSVPanel(wx.Panel):
    
    """
     glinit must be called first
    """
    def __init__(self,parent,**kwargs):
        super().__init__(parent,name="JUMEG_TSV_PNL") #,ShowTitleB=False)
        self._IO_DATA  = JuMEG_TSV_Utils_IO_Data()
        self._PNL_PLOT = None
        self._wx_init(**kwargs)
        self._ApplyLayout()
    
    @property
    def IOdata(self): return self._IO_DATA
    @property
    def PlotPanel(self): return self._PNL_PLOT

    def _update_from_kwargs(self,**kwargs):
       #--- io
        self.fname      = kwargs.get("fname")
        self.path       = kwargs.get("path")
        self.bads       = kwargs.get("bads")
        # self.experiment = kwargs.get("experiment")
       #---
        self.verbose    = kwargs.get("verbose")
        self.debug      = kwargs.get("debug")
        
        if self.PlotPanel:
           self.PlotPanel._update_from_kwargs(**kwargs)
        
    def __load_data(self,**kwargs):
        msg = "Please wait ... loading file: {}\n path: {}".format(kwargs.get("fname"),path=kwargs.get("path") )
        
        msg = "Please wait ..."
        busy = PBI.PyBusyInfo(msg, parent=None, title="JuMEG TSV loading data")
  
       # self.IOdata.update(**kwargs)

       
        del busy

    def _wx_init(self,**kwargs):
        self._update_from_kwargs(**kwargs)
        self._PNL_PLOT = JuMEG_TSV_wxPlot2D(self,**kwargs)
        self.update_parameter(**kwargs)
       
   #---
    def update_parameter(self,**kwargs):
        """
        update argparser default parameter from template
        set choices in BTiCTLs, set value to first item
        """
        self.IOdata.update(**kwargs)
        
        #self.update_data(**kwargs)  #raw=fname=self.fname,n_channels=self.n_channels,cols=self.n_cols)

   #--- data update
    def update_data(self,**kwargs):
        pub.sendMessage("MAIN_FRAME.BUSY",value=True)
        if kwargs:
           self.IOdata.update(**kwargs)
           pub.sendMessage("MAIN_FRAME.UPDATE_BADS",value="LOADED")
           
        if self.IOdata.isLoaded:
           self.PlotPanel.update(raw=self.IOdata.raw) #,**kwargs)
           self.GetParent().StatusBar.SetStatusText(self.IOdata.path,1)
           self.GetParent().StatusBar.SetStatusText(self.IOdata.fname,2)

        pub.sendMessage("MAIN_FRAME.BUSY",value=False)
        
    def update_on_display(self):
        #self.SplitterAB.SetSashPosition(self.GetSize()[0] / 2.0,redraw=True)
        pass
      
    def init_pubsub(self,**kwargs):
        """ init pubsub call overwrite """
        #pub.subscribe(self.ClickOnExperimentTemplateUpdate,self.ExpTemplate.GetMessage("UPDATE"))
        pass
    
    def ShowPlotSettings(self,type="subplot"):
        """
        ToDo plot setting
        type: subplot, waterfall,sensorlayout
        combbox
        
        :param type:
        :return:
        """
        opt = self.PlotPanel.GetPlotOptions()
        param={ "plot": {"counts": opt.channels.counts,"n_plots":opt.n_plots,"n_cols":opt.n_cols,"start":opt.plot_start},
                "time": {"pretime":opt.time.pretime,"start":opt.time.start,"end":opt.time.end,"window":opt.time.window,"inc_factor":opt.time.inc_factor,"scroll_step":opt.time.scroll_step}
               }
        param = jumeg_tsv_wxutils_dlg_plot_settings(**param,type=type)
       
        if param:
          # logger.info("Plot PARAM: {}".format(param))
           self.PlotPanel.update(**param)
    
    def ShowGroupSettings(self,evt=None):
        '''
        ok=ShowGroupDialog( self.PlotPanel.GetGroupParameter() )
        if ok:
          self.PlotPanel.SetGroupParameter( ShowGroupDialog.GetGroupParameter() )
          self.PlotPanel.update_group_option(raw_is_loaded=self.IOdata.raw_is_loaded)
        :param evt:
        :return:
        '''
        grp = deepcopy( self.PlotPanel.GetGroupSettings() )
        
        dlg = GroupDLG(self,grp=grp )
        out = dlg.ShowModal()
        if out == wx.ID_APPLY:
           self.PlotPanel.update(settings=grp)
           
           #grp = dlg.Group.GetGroup(None)
           #for g in grp.keys():
           #    logger.info("OUTPUT: {} => {} \n {}".format(g,dlg.Group.GetScaling(g),grp.get(g)))
           #    dlg.Destroy()
        
        #opt = self.PlotPanel.GetSettings()
        # self.PlotPanel.update_group_option(raw_is_loaded=self.IOdata.raw_is_loaded)
        #param = self.PlotPanel.plot.plot.data.info
        # self.PlotPanel.plot.plot.data.info.GetInfo()
        #param = jumeg_tsv_wxutils_dlg_group_settings(opt=param)
    
    def ClickOnCancel(self,evt):
        wx.LogMessage("Click <Cancel> button")
        #wx.CallAfter(pub.sendMessage,"MAIN_FRAME.MSG.INFO",data="<Cancel> button is no in use")
        #self.SubProcess.Cancel()
    
    def ClickOnCtrls(self,evt):
        obj = evt.GetEventObject()
        #print("\n ---> ClickOnCTRL:".format( self.GetName() ))
        #print("OBJ Name => "+ obj.GetName() )
        
        if obj.GetName() == "TEST":
           self.update_data()
        #else:
        #    evt.Skip()
        
    def ShowOpenFileDialog(self):
        fname = jumeg_tsv_wxutils_openfile(self,path=self.IOdata.path)
        if fname:
           path  = os.path.dirname(fname)
           fname = os.path.basename(fname)
           self.update_data(fname=fname,path=path,reload=False)
    
    def ShowSaveFileDialog(self):
        pass
        # fname = jumeg_tsv_wxutils_savefile(self,path=self.IOdata.path,)
        
    def SaveBads(self):
         if self.IOdata.save():
            pub.sendMessage("MAIN_FRAME.UPDATE_BADS",value="SAVED")
         else:
            pub.sendMessage("MAIN_FRAME.UPDATE_BADS",value="UPDATE")
    
    def _ApplyLayout(self):
        LEA = wx.LEFT|wx.EXPAND|wx.ALL
        vbox = wx.BoxSizer(wx.VERTICAL)
        #vbox.Add(self._bt,0,LEA,2)
        vbox.Add(self._PNL_PLOT,1,LEA,2)
    
       #--- fix size to show combos of scale and unit
        #stl = wx.StaticLine(self,size=(350,2) )
        #stl.SetBackgroundColour("GREY85")
        #vbox.Add(stl,0,LEA,1)
        #vbox.Add(self._pnl_button_box,0,LEA,2)
    
        self.SetAutoLayout(True)
        self.SetSizer(vbox)
        self.Fit()
        self.Layout()


class JuMEG_GUI_TSVFrame(wx.Frame):
    __slots__=[ "_STB","_TB","_PLT","_verbose","_debug","_menu_info","_menu_help","_wx_file_combobox","_bt_save"]
    def __init__(self,parent,id,title,pos=wx.DefaultPosition,size=[1024,768],name='JuMEGTSV',*kargs,**kwargs):
        style = wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE
        super().__init__(parent,id,title,pos,size,style,name) #,**kwargs)
       #--- wx
        self._STB     = None
        self._TB      = None
        self._PLT     = None
        self._menu_help=None
        self._menu_info=None
        self._wx_file_combobox = None
        self._bt_save = None
      #  self.AboutBox = JuMEG_wxAboutBox()
       #--- flags
        self._debug   = False
        self._verbose = False
        
        self._wx_init(**kwargs)
        self._init_pubsub()
        
    @property
    def FileListBox(self): return self._wx_file_combobox
    @property
    def PlotPanel(self): return self._PLT
    @property
    def StatusBar(self): return self._STB
    @property
    def ToolBar(self):   return self._TB
    @property
    def BtSave(self): return self._bt_save
    
    @property
    def debug(self): return self._debug
    @debug.setter
    def debug(self,v):
        if self._PLT:
           self._PLT.debug=v
        self._debug=v
        if v:
           warnings.filterwarnings("ignore")
        else:
           warnings.filterwarnings("default")
    
    
    @property
    def verbose(self): return self._verbose
    @verbose.setter
    def verbose(self,v):
        if self._PLT:
           self._PLT.verbose=v
        self._verbose=v
        

    def _update_from_kwargs(self,**kwargs):
        self._debug   = kwargs.get("debug",False)
        self._verbose = kwargs.get("verbose",False)

    def _wx_init(self,**kwargs):
        w,h = wx.GetDisplaySize()
        self.SetSize(w/1.1,h/1.3)
        self.Center()
        
        self._update_from_kwargs(**kwargs)
        self._wx_init_menu()
        self._wxInitStatusBar(fields=5) #  msg,path ,fname , n_bads, status
        self._wxInitToolBar()
        self._PLT = JuMEG_wxTSVPanel(self,**kwargs)
       
        return self._PLT
    
    def _wxInitToolBar(self,**kwargs):
        """
      
        """
        style=wx.TB_RIGHT
        style=wx.TB_HORIZONTAL
       # style=wx.TB_BOTTOM
        
        tsize=[24,24]
        tb =  self.CreateToolBar(style=style)
        tb.SetToolBitmapSize(tsize)  # sets icon size
      
        ctrls=[
               ["Load","Load Data",wx.ART_FILE_OPEN,self.ClickOnLoadFile],
              # ["Save","Save Bads",wx.ART_FILE_SAVE,self.ClickOnSaveFile],
              # ["Save As","Save As",wx.ART_FILE_SAVE_AS,self.ClickOnSaveFileAs],
              ]
       
        for ctrl in ctrls:
            if len(ctrl):
               bmp  = wx.ArtProvider.GetBitmap(ctrl[2],wx.ART_TOOLBAR,tsize)
               tool = tb.AddTool(-1, ctrl[0],bmp, wx.NullBitmap, wx.ITEM_NORMAL,ctrl[1],ctrl[1])
               self.Bind(wx.EVT_TOOL, ctrl[-1], tool)
            else:
               tb.AddSeparator()
        
       #--- use save button
        tb.AddSeparator()
        self._bt_save = wx.Button(tb,-1,"S A V E",name="TB.BT.SETTINGS.PLOT")
        self._bt_save.SetToolTip("save raw obj with bad channels to disk")
        tool = tb.AddControl(self._bt_save,label="SAVE Bads")
        self.Bind(wx.EVT_BUTTON,self.ClickOnSaveFile,tool)
        tb.AddSeparator()




         #btn = wx.BitmapButton(tb, wx.ID_ANY, make_bitmap(wx.Bitmap(gifpath)), wx.DefaultPosition, (32, 32), wx.BU_AUTODRAW|wx.RAISED_BORDER)
      
      #---
        ctrls=[
               ["Load List","Load File List",wx.ART_FOLDER_OPEN,self.ClickOnLoadFilesFromList],
              ]
        self._wx_file_combobox = wx.ComboBox(tb,-1,"File List",choices=["","This","is a","wx.ComboBox"],size=(-1,-1),style=wx.CB_DROPDOWN|wx.EXPAND )
        tb.AddSeparator()
        tb.AddControl(self._wx_file_combobox)
        
        for ctrl in ctrls:
            if len(ctrl):
               bmp  = wx.ArtProvider.GetBitmap(ctrl[2],wx.ART_TOOLBAR,tsize)
               tool = tb.AddTool(-1, ctrl[0],bmp, wx.NullBitmap, wx.ITEM_NORMAL,ctrl[1],ctrl[1])
               self.Bind(wx.EVT_TOOL, ctrl[-1], tool)
            else:
               tb.AddSeparator()
        #self.Bind(wx.EVT_COMBOBOX,self.OnCombo,id=cbID)
        tb.AddSeparator()
        
       #--- group plot options
        tb.AddStretchableSpace()
        #tb.AddControl(wx.StaticText(tb,-1,"Settings"))
        bt = wx.Button(tb,-1,"P L O T ` s",name="TB.BT.SETTINGS.PLOT")
        bt.SetToolTip("change plot settings")
        tool = tb.AddControl(bt,label="PlotSettings")
        self.Bind(wx.EVT_BUTTON, self.ClickOnButton, tool)
        
        bt=wx.Button(tb,-1,"G R O U P `s",name="TB.BT.SETTINGS.GROUP")
        bt.SetToolTip("change group settings")
        tool = tb.AddControl(bt,label="GroupSettings")
        self.Bind(wx.EVT_BUTTON,self.ClickOnButton,tool)
        
        
        if self.debug:
           # tb.AddStretchableSpace()
           tb.AddSeparator()
           tool = tb.AddControl(wx.Button(tb,-1,"DEMO",name="TB.BT.DEMO"),label="TEST")
           self.Bind(wx.EVT_BUTTON, self.ClickOnButton, tool)
        
        tb.AddSeparator()
        
       #--- end  set Help
        ctrls = [
                  ["Misc","Misc",wx.ART_INFORMATION,self._menu_info],
                  ["Help","Help",wx.ART_HELP,self._menu_help],
                ]
        for ctrl in ctrls:
            if len(ctrl):
               bmp = wx.ArtProvider.GetBitmap(ctrl[2],wx.ART_TOOLBAR,tsize)
               tool = tb.AddTool(-1,ctrl[0],bmp,wx.NullBitmap,wx.ITEM_NORMAL,ctrl[1],ctrl[1])
               tool.SetDropdownMenu( ctrl[-1] )
               #self.Bind(wx.EVT_TOOL,ctrl[-1],tool)
            else:
               tb.AddSeparator()
     
        tb.Realize()
        self._TB = tb
        
        # self.Bind(wx.EVT_TOOL,self.ClickOnToolBar)
    def ClickOnToolBar(self,evt):
        pass
    
    def ClickOnAbout(self,evt):
        self.AboutBox.show(self)

    def _wxInitStatusBar(self,fields=5,w=[80,-2,-1.5,-1,100],status_style=wx.SB_SUNKEN):
        if self._STB:
           self._STB.Destroy()
        self._STB = self.CreateStatusBar(fields,style=wx.STB_DEFAULT_STYLE)
        self._STB.SetStatusWidths(w) # 200 pixel,66%,33%,200 pix
        st=[]
        for i in range( len(w)):
            st.append(status_style)
        self._STB.SetStatusStyles(st)

    def SetStatusBarMSG(self,data):
        '''
         call from pubsub
         "MAIN_FRAME.STB.MSG", value=["RUN", self._args[0], "PID", str(self.__proc.pid)]
        '''
        idx = 0
        if not data: return
        if not isinstance(data,(list)): data = list(data)
        if data:
            for s in data:
                #self.msg_info("STB: "+s +"  "+str(idx))
                self._STB.SetStatusText(s,i=idx)
                idx += 1
                if idx >= self._STB.GetFieldsCount(): break

    def OnExitApp(self,evt):
        self.ClickOnClose(evt)

    def ToggleCheckBox(self,evt):
        obj = evt.GetEventObject()
        v   = obj.GetValue()
        name= obj.GetName()
        if name =="StatusBar":
           if v:
               self.StatusBar.Show()
           else:
               self.StatusBar.Hide()
           return
        elif name =="Verbose":
             self.verbose = v
        elif name =="Debug":
             self.debug=v
        
        if self.use_pubsub:  # verbose,debug status
           pub.sendMessage('MAIN_FRAME.' + name.upper(), value=v)


    def ClickOnButton(self,evt):
        obj = evt.GetEventObject()
        if obj.GetName() == "TB.BT.DEMO":
           self.PlotPanel.update_data()
        elif obj.GetName() == "TB.BT.SETTINGS.PLOT":
           self.PlotPanel.ShowPlotSettings()
        elif obj.GetName() == "TB.BT.SETTINGS.GROUP":
           self.PlotPanel.ShowGroupSettings()

        else:
          evt.Skip()
        
    def Show(self,show=True):
       #--- make sure  Frame is on screen and OGL is init
        super().Show(show=show)
        
        try:
            self.PlotPanel.update_data()
        except:
            pass
        
        
    def AboutBox(self):
        self.AboutBox.description = "Time Series Viewer for MEG/EEG data in FIF format"
        self.AboutBox.version = __version__
        self.AboutBox.copyright = '(C) 2019 Frank Boers <f.boers@fz-juelich.de>'
        self.AboutBox.developer = 'Frank Boers'
        self.AboutBox.docwriter = 'Frank Boers'

    def ClickOnLoadFile(self,evt):
        self.PlotPanel.ShowOpenFileDialog()
      
    def ClickOnSaveFileAs(self,evt):
        self.PlotPanel.ShowSaveFileDialog()

    def ClickOnSaveFile(self,evt):
        cl = self.BtSave.GetBackgroundColour()
        self.BtSave.SetBackgroundColour(wx.YELLOW)
        self.PlotPanel.SaveBads()
        self.BtSave.SetBackgroundColour(cl)
        
    def ClickOnLoadFilesFromList(self,evt):
        pass
  
    
    def ClickOnClose(self,evt):
       # if not self.debug:
        if wx.MessageBox('Are you sure to quit?',"Please confirm", wx.ICON_QUESTION | wx.YES_NO) != wx.YES:
              return
        self.Destroy()
        evt.Skip()
  
    def msg_error(self,data="ERROR"):
        if isinstance(data,(list)):
           data= "\n".join(data)
        wx.MessageBox("Error: " + data, caption="ERROR  " + self.Name, style=wx.ICON_ERROR | wx.OK)

    def msg_warning(self,data="WARNING"):
        if isinstance(data, (list)):
           data = "\n".join(data)
        wx.MessageBox("Warning: "+data,caption="Warning  " +self.Name,style=wx.ICON_Warning|wx.OK)

    def msg_info(self,data="INFO"):
        if isinstance(data,(list)):
           msgtxt= "\n".join(data)
        wx.MessageBox("Info: "+data,caption="Info  " +self.Name,style=wx.ICON_INFORMATION|wx.OK)

    def _wx_init_menu(self):
        self._menu_info = wx.Menu()
        itm1 = self._menu_info.Append(wx.ID_ANY,"&Verbose",'set verbose',wx.ITEM_CHECK)
        itm2 = self._menu_info.Append(wx.ID_ANY,"&Debug",'set debug',wx.ITEM_CHECK)

        #self.Bind(wx.EVT_MENU,lambda evt,label=l:self.ClickOnLogger(evt,label),itm1)
        #self.Bind(wx.EVT_MENU,lambda evt,label=l:self.ClickOnLogger(evt,label),itm2)

        self._menu_help = wx.Menu()
        itm1 = self._menu_help.Append(wx.ID_ANY,"&About",'show About',wx.ITEM_NORMAL)
        itm2 = self._menu_help.Append(wx.ID_ANY,"&Help",'show help',wx.ITEM_NORMAL)

    def _init_pubsub(self):
        """ init pubsub call and messages"""
       #---
        pub.subscribe(self.msg_error,  "MAIN_FRAME.MSG.ERROR")
        pub.subscribe(self.msg_warning,"MAIN_FRAME.MSG.WARNING")
        pub.subscribe(self.msg_info,   "MAIN_FRAME.MSG.INFO")
       #---
        pub.subscribe(self.SetStatusBarMSG, "MAIN_FRAME.STB.MSG")
        pub.subscribe(self.ClickOnClose, "MAIN_FRAME.CLICK_ON_CLOSE")
       #---
        pub.subscribe(self.wxUpdateBusy,"MAIN_FRAME.BUSY")
       # --- BADS
        pub.subscribe(self.wxUpdateBads,"MAIN_FRAME.UPDATE_BADS")
       #--- verbose debug
        pub.sendMessage('MAIN_FRAME.VERBOSE', value=self.verbose)
        pub.sendMessage('MAIN_FRAME.DEBUG', value=self.debug)
    
    def wxUpdateBusy(self,value=False):
        pass
        
    def wxUpdateBads(self,value=""):
         '''
         call from pubsub "MAIN_FRAME.UPDATE_BADS"
         '''
         self._STB.SetStatusText(value,i=0)
         bads = self.PlotPanel.IOdata.GetBads()
         self._STB.SetStatusText(str( len(bads) ),i=4)
         if bads:
            self._STB.SetStatusText(",".join(bads),i=3)
            
         if value=="CHANGED":
            self._STB.SetBackgroundColour('YELLOW')
         else:
            self._STB.SetBackgroundColour('#E0E2EB')
            
         self._STB.Refresh()
         
        
#----
def get_args(argv):
    # ToDO: make argparser  sub cls
    info_global = """
     JuMEG Time Series Viewer [TSV] Start Parameter

     ---> view time series data FIF file
      jumeg_gui_tsv01.py --fname=110058_MEG94T_121001_1331_1_c,rfDC-raw.fif --path=/localdata/frank/data/MEG94T/mne/110058/MEG94T/121001_1331/1 -v

      jumeg_gui_tsv01.py --fname=110058_MEG94T_121001_1331_1_c,rfDC_30sec-raw.fif --path=/localdata/frank/data/MEG94T/mne/110058/MEG94T/121001_1331/1 -v

    """

    Hdri_prime= """
                which graphiccard to select
                https://askubuntu.com/questions/1038271/intel-amd-hybrid-graphics-ubuntu-18-04
                xrandr --listproviders
                xrandr --setprovideroffloadsink 1 0

                DRI_PRIME=0 glxinfo | grep "OpenGL renderer"
                DRI_PRIME=1 glxinfo | grep "OpenGL renderer"

                DRI_PRIME=0 glmark2 --fullscreen
                DRI_PRIME=1 glmark2 --fullscreen
                """
    # parser = argparse.ArgumentParser(description='JuMEG TSV Start Parameter')
    
    parser = argparse.ArgumentParser(info_global)
    
    parser.add_argument("-p","--path",help="start path to data files")
    parser.add_argument("-f","--fname",help="Input raw FIF file")
    
    parser.add_argument("-exp","--experiment",help="experiment name")
    # parser.add_argument("-bads","--bads",help="list of channels to mark as bad --bads=MEG 007,MEG 142,MEG 156,RFM 011")
    parser.add_argument("-v","--verbose",action="store_true",help="verbose mode")
    parser.add_argument("-d","--debug",action="store_true",help="debug mode")
    parser.add_argument("-dd","--dd",action="store_true",help="dd-debug mode")
    parser.add_argument("-ddd","--ddd",action="store_true",help="ddd-debug mode")
    parser.add_argument("-log","--logfile",action="store_true",help="generate logfile")
    
   #--- mne opt
    parser.add_argument("-tw","--window",type=float,help="Time window for plotting (sec)",default=120.0)
    parser.add_argument("-t0","--start",type=float,help="Initial start time for ploting",default=0.0)
    parser.add_argument("-nch","--n_channels",type=int,help="Number of channels to plot",default=40)
    parser.add_argument("-nco","--n_cols",type=int,help="Number of cols to plot",default=1)
    parser.add_argument("-dri","--dri_prime",type=int,help="ENV for selecting grahiccard",default=0)
    #--- plot option
    # parser.add_argument("--subplot",    type=int,  help="Subplot configuration --subplot=3,2 ")
    
    #--- init flags
    # ck if flag is set in argv as True
    # problem can not switch on/off flag via cmd call
    opt = parser.parse_args()
    for g in parser._action_groups:
        for obj in g._group_actions:
            if str(type(obj)).endswith('_StoreTrueAction\'>'):
                if vars(opt).get(obj.dest):
                    opt.__dict__[obj.dest] = False
                    for flg in argv:
                        if flg in obj.option_strings:
                           opt.__dict__[obj.dest] = True
                           break
    
    return opt,parser


def DisplayInfo():
    pp = wx.GetDisplayPPI()
    size_pix = wx.GetDisplaySize()
    size_mm = wx.DisplaySizeMM()
    logger.info("---> Display Info: pixel size: {} mm: ~{} ? pp: {}".format(size_pix,size_mm,pp))
    

#---
def run(opt):
   
    if opt.debug:
       opt.verbose = True
    #    opt.path = "./data/"
    #    opt.fname =  '007_test_10_raw.fif'
    elif opt.dd:
        opt.path    = "$JUMEG_TEST_DATA/data/"
        opt.fname   = '007_test_60_raw.fif'
        opt.verbose = True
        opt.debug   = True
    elif opt.ddd:
        opt.path    = "$JUMEG_TEST_DATA/data/"
        opt.fname   = "007_test_c,rfDC,meeg,nr,bcc-raw.fif"
        opt.verbose = True
        opt.debug   = True
        
    app = wx.App()
    frame = JuMEG_GUI_TSVFrame(None,-1,'JuMEG TSV',
                               fname=opt.fname,path=opt.path,verbose=opt.verbose,debug=opt.debug,experiment=opt.experiment,
                               duration=opt.window,start=opt.start,n_channels=opt.n_channels,n_cols=opt.n_cols) #,bads=opt.bads
                               
    DisplayInfo()
    frame.Show()
    
    app.MainLoop()
#=========================================================================================
#==== MAIN
#=========================================================================================
if __name__ == "__main__":
    opt,parser = get_args(sys.argv)
    
   #--- select 2.graphic card e.g. notebook
    os.putenv("DRI_PRIME",str(opt.dri_prime))

    jumeg_logger.setup_script_logging(name=sys.argv[0],opt=opt,logger=logger)
    
    run(opt)


