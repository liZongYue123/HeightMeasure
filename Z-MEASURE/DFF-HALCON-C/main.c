/*****************************************************************************
 * File generated by HDevelop for HALCON/C Version 23.11.0.0
 * Non-ASCII strings in this file are encoded in local-8-bit encoding (cp936).
 * Ensure that the interface encoding is set to locale encoding by calling
 * SetHcInterfaceStringEncodingIsUtf8(FALSE) at the beginning of the program.
 * 
 * Please note that non-ASCII characters in string constants are exported
 * as octal codes in order to guarantee that the strings are correctly
 * created on all systems, independent on any compiler settings.
 * 
 * Source files with different encoding should not be mixed in one project.
 *****************************************************************************/
#include "HalconC.h"
#include "halconc/Hdevthread.h"


/* Procedure declarations */
/* External procedures */
/* Chapter: Develop*/
/* Short Description: Switch dev_update_pc, dev_update_var, and dev_update_window to 'off'. */
extern void dev_update_off ();
/* Chapter: Graphics / Text*/
/* Short Description: Write one or multiple text messages. */
extern void disp_message (Htuple hv_WindowHandle, Htuple hv_String, Htuple hv_CoordSystem, 
    Htuple hv_Row, Htuple hv_Column, Htuple hv_Color, Htuple hv_Box);
/* Chapter: File / Misc*/
/* Short Description: Get all image files under the given path */
extern void list_image_files (Htuple hv_ImageDirectory, Htuple hv_Extensions, Htuple hv_Options, 
    Htuple *hv_ImageFiles);
/* Chapter: Graphics / Text*/
/* Short Description: Set font independent of OS */
extern void set_display_font (Htuple hv_WindowHandle, Htuple hv_Size, Htuple hv_Font, 
    Htuple hv_Bold, Htuple hv_Slant);

/* Procedures */
#ifndef NO_EXPORT_MAIN
/* Main procedure */
void action()
{
  /* Stack for temporary tuples */
  Htuple   TTemp[100];
  int      SP=0;
  /* Stack for temporary tuple vectors */
  Hvector  TVTemp[100] = {0};
  int      SPTV=0;

  /* Local iconic variables */
  Hobject  ho_ImageArray, ho_Image, ho_ImageDisp;
  Hobject  ho_Depth, ho_Confidence, ho_DepthScaleMax, ho_Sharp;
  Hobject  ho_ImageScaled, ho_MultiChannelImage;

  /* Local control variables */
  Htuple  hv_WindowHandle, hv_Names, hv_Index;

  /* Initialize iconic variables */
  gen_empty_obj(&ho_ImageArray);
  gen_empty_obj(&ho_Image);
  gen_empty_obj(&ho_ImageDisp);
  gen_empty_obj(&ho_Depth);
  gen_empty_obj(&ho_Confidence);
  gen_empty_obj(&ho_DepthScaleMax);
  gen_empty_obj(&ho_Sharp);
  gen_empty_obj(&ho_ImageScaled);
  gen_empty_obj(&ho_MultiChannelImage);

  /* Initialize control variables */
  create_tuple(&hv_WindowHandle,0);
  create_tuple(&hv_Names,0);
  create_tuple(&hv_Index,0);

  /****************************************************/
  /******************   Begin procedure   *************/
  /****************************************************/

  /*dev_update_off ()*/
  /***/dev_update_off();

  if (hdev_window_stack_is_open())
  {
    Htuple active_win;
    create_tuple(&active_win,0);
    hdev_window_stack_pop(&active_win);
    T_close_window(active_win);
    destroy_tuple(active_win);
  }
  if (hdev_window_stack_is_open())
  {
    destroy_tuple(hv_WindowHandle);
    hdev_window_stack_get_active(&hv_WindowHandle);
  }
  /*set_display_font (WindowHandle, 16, 'mono', 'true', 'false')*/
  create_tuple_i(&TTemp[SP++],16);
  create_tuple_s(&TTemp[SP++],"mono");
  create_tuple_s(&TTemp[SP++],"true");
  create_tuple_s(&TTemp[SP++],"false");
  /***/set_display_font(hv_WindowHandle, TTemp[SP-4], TTemp[SP-3], TTemp[SP-2], TTemp[SP-1]);
  destroy_tuple(TTemp[--SP]);
  destroy_tuple(TTemp[--SP]);
  destroy_tuple(TTemp[--SP]);
  destroy_tuple(TTemp[--SP]);

  if (hdev_window_stack_is_open())
  {
    Htuple active_win;
    create_tuple(&active_win,0);
    hdev_window_stack_get_active(&active_win);
    /*dev_set_paint ('default')*/
    create_tuple_s(&TTemp[SP++],"default");
    /***/T_set_paint(active_win,TTemp[SP-1]);
    destroy_tuple(TTemp[--SP]);
    destroy_tuple(active_win);
  }
  /*Read the sequence of images*/
  /*disp_message (WindowHandle, 'Read a sequence of focus images ...', 'image', 20, 20, 'white', 'false')*/
  create_tuple_s(&TTemp[SP++],"Read a sequence of focus images ...");
  create_tuple_s(&TTemp[SP++],"image");
  create_tuple_i(&TTemp[SP++],20);
  create_tuple_i(&TTemp[SP++],20);
  create_tuple_s(&TTemp[SP++],"white");
  create_tuple_s(&TTemp[SP++],"false");
  /***/disp_message(hv_WindowHandle, TTemp[SP-6], TTemp[SP-5], TTemp[SP-4], TTemp[SP-3], 
      TTemp[SP-2], TTemp[SP-1]);
  destroy_tuple(TTemp[--SP]);
  destroy_tuple(TTemp[--SP]);
  destroy_tuple(TTemp[--SP]);
  destroy_tuple(TTemp[--SP]);
  destroy_tuple(TTemp[--SP]);
  destroy_tuple(TTemp[--SP]);

  /*list_image_files ('F:/3D/DepthFromFocus/DepthFromFocus/Origin', 'default', [], Names)*/
  create_tuple_s(&TTemp[SP++],"F:/3D/DepthFromFocus/DepthFromFocus/Origin");
  create_tuple_s(&TTemp[SP++],"default");
  create_tuple(&TTemp[SP++],0);
  destroy_tuple(hv_Names);
  /***/list_image_files(TTemp[SP-3], TTemp[SP-2], TTemp[SP-1], &hv_Names);
  destroy_tuple(TTemp[--SP]);
  destroy_tuple(TTemp[--SP]);
  destroy_tuple(TTemp[--SP]);

  /*read_image (ImageArray, Names)*/
  clear_obj(ho_ImageArray);
  /***/T_read_image(&ho_ImageArray, hv_Names);

  /*channels_to_image (ImageArray, Image)*/
  clear_obj(ho_Image);
  /***/channels_to_image(ho_ImageArray, &ho_Image);

  /* stop(...); only in hdevelop*/
  /*========== for Index := 20 to 107 by 1 ==========*/
  create_tuple_i(&TTemp[SP++],107);
  create_tuple_i(&TTemp[SP++],1);
  create_tuple_i(&TTemp[SP++],20);
  T_tuple_greater(TTemp[SP-1],TTemp[SP-3],&TTemp[SP]);
  SP++;
  T_tuple_equal(TTemp[SP-2],TTemp[SP-4],&TTemp[SP]);
  if(get_i(TTemp[SP],0) ||
     (!((( get_i(TTemp[SP-1],0)) && (get_d(TTemp[SP-3],0)>0)) ||
        ((!get_i(TTemp[SP-1],0)) && (get_d(TTemp[SP-3],0)<0)))))
  {
   destroy_tuple(TTemp[SP--]);
   destroy_tuple(TTemp[SP]);
   T_tuple_sub(TTemp[SP-1],TTemp[SP-2],&TTemp[SP]);
   destroy_tuple(hv_Index);
   copy_tuple(TTemp[SP],&hv_Index);
   destroy_tuple(TTemp[SP--]);
   destroy_tuple(TTemp[SP]);
   for(;;)
   {
   T_tuple_add(hv_Index,TTemp[SP-1],&TTemp[SP]);
   destroy_tuple(hv_Index);
   copy_tuple(TTemp[SP],&hv_Index);
   destroy_tuple(TTemp[SP]);
   if(get_d(TTemp[SP-1],0)<0)
    T_tuple_less(hv_Index,TTemp[SP-2],&TTemp[SP]);
   else
    T_tuple_greater(hv_Index,TTemp[SP-2],&TTemp[SP]);
   if(get_i(TTemp[SP],0)) break;
   destroy_tuple(TTemp[SP]);
   /*========== for ==========*/

    /*access_channel (Image, ImageDisp, Index)*/
    clear_obj(ho_ImageDisp);
    /***/T_access_channel(ho_Image, &ho_ImageDisp, hv_Index);

    if (hdev_window_stack_is_open())
    {
      Htuple active_win;
      create_tuple(&active_win,0);
      hdev_window_stack_get_active(&active_win);
      /*dev_display (ImageDisp)*/
      /***/T_disp_obj(ho_ImageDisp, active_win);
      destroy_tuple(active_win);
    }
    /*wait_seconds (0.1)*/
    /***/wait_seconds(0.1);

   }
   destroy_tuple(TTemp[SP--]);
   destroy_tuple(TTemp[SP--]);
   destroy_tuple(TTemp[SP]);
  }
  else
  {
   destroy_tuple(TTemp[SP--]);
   destroy_tuple(TTemp[SP--]);
   destroy_tuple(TTemp[SP--]);
   destroy_tuple(TTemp[SP--]);
   destroy_tuple(TTemp[SP]);
  }/*========== end for ========*/

  /*Compute the depth map and display results*/
  /*dev_display (ImageDisp)*/
  /*disp_message (WindowHandle, 'Compute the depth map', 'image', 20, 20, 'white', 'false')*/
  /*depth_from_focus (Image, Depth, Confidence, ['bandpass', 3, 3], 'next_maximum')*/
  create_tuple(&TTemp[SP++],3);
  set_s(TTemp[SP-1],"bandpass",0);
  set_i(TTemp[SP-1],3  ,1);
  set_i(TTemp[SP-1],3  ,2);
  create_tuple_s(&TTemp[SP++],"next_maximum");
  clear_obj(ho_Depth);
  clear_obj(ho_Confidence);
  /***/T_depth_from_focus(ho_Image, &ho_Depth, &ho_Confidence, TTemp[SP-2], TTemp[SP-1]);
  destroy_tuple(TTemp[--SP]);
  destroy_tuple(TTemp[--SP]);

  /*scale_image_max (Depth, DepthScaleMax)*/
  clear_obj(ho_DepthScaleMax);
  /***/scale_image_max(ho_Depth, &ho_DepthScaleMax);

  /*select_grayvalues_from_channels (Image, Depth, Sharp)*/
  clear_obj(ho_Sharp);
  /***/select_grayvalues_from_channels(ho_Image, ho_Depth, &ho_Sharp);

  /*scale_image (Sharp, ImageScaled, 8, 0)*/
  clear_obj(ho_ImageScaled);
  /***/scale_image(ho_Sharp, &ho_ImageScaled, 8, 0);

  /*compose2 (DepthScaleMax, ImageScaled, MultiChannelImage)*/
  clear_obj(ho_MultiChannelImage);
  /***/compose2(ho_DepthScaleMax, ho_ImageScaled, &ho_MultiChannelImage);

  if (hdev_window_stack_is_open())
  {
    Htuple active_win;
    create_tuple(&active_win,0);
    hdev_window_stack_get_active(&active_win);
    /*dev_clear_window ()*/
    /***/T_clear_window(active_win);
    destroy_tuple(active_win);
  }
  /*dev_set_paint (['3d_plot', 'texture'])*/
  /*dev_display (MultiChannelImage)*/
  /*disp_message (WindowHandle, '3D reconstruction of IGBT', 'image', 20, 280, 'white', 'false')*/

  /****************************************************/
  /******************     End procedure   *************/
  /****************************************************/

  /* Clear temporary tuple stack */
  while (SP > 0)
    destroy_tuple(TTemp[--SP]);
  /* Clear temporary tuple vectors stack*/
  while (SPTV > 0)
    V_destroy_vector(TVTemp[--SPTV]);
  /* Clear local iconic variables */
  clear_obj(ho_ImageArray);
  clear_obj(ho_Image);
  clear_obj(ho_ImageDisp);
  clear_obj(ho_Depth);
  clear_obj(ho_Confidence);
  clear_obj(ho_DepthScaleMax);
  clear_obj(ho_Sharp);
  clear_obj(ho_ImageScaled);
  clear_obj(ho_MultiChannelImage);

  /* Clear local control variables */
  destroy_tuple(hv_WindowHandle);
  destroy_tuple(hv_Names);
  destroy_tuple(hv_Index);

}


#ifndef NO_EXPORT_APP_MAIN

int main(int argc, char *argv[])
{
  /* Default settings used in HDevelop */
  int ret=0;
  Htuple Parameter, Value;

#if defined(_WIN32)
  set_system("use_window_thread", "true");
#endif

  /* file was stored with local-8-bit encoding
   *   -> set the interface encoding accordingly */
  SetHcInterfaceStringEncodingIsUtf8(FALSE);

  create_tuple(&Parameter,2);
  create_tuple(&Value,2);
  set_s(Parameter,"width",0);
  set_i(Value,512,0);
  set_s(Parameter,"height",1);
  set_i(Value,512,1);
  T_set_system(Parameter,Value);
  destroy_tuple(Value);
  destroy_tuple(Parameter);

  action();

#if defined(_WIN32)
  /*
   * On Windows socket communication is no longer possible after returning
   * from main, so HALCON cannot return floating licenses automatically.
   */
  set_system("return_license", "true");
#endif

  return ret;
}

#endif


#endif


