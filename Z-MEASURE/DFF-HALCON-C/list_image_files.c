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



/* Chapter: File / Misc*/
/* Short Description: Get all image files under the given path */
void list_image_files (Htuple hv_ImageDirectory, Htuple hv_Extensions, Htuple hv_Options, 
    Htuple *hv_ImageFiles)
{

  /* Stack for temporary tuples */
  Htuple   TTemp[100];
  int      SP=0;
  /* Stack for temporary tuple vectors */
  Hvector  TVTemp[100] = {0};
  int      SPTV=0;

  /* Local iconic variables */

  /* Local control variables */
  Htuple  hv_ImageDirectoryIndex, hv_ImageFilesTmp;
  Htuple  hv_CurrentImageDirectory, hv_HalconImages, hv_OS;
  Htuple  hv_Directories, hv_Index, hv_Length, hv_NetworkDrive;
  Htuple  hv_Substring, hv_FileExists, hv_AllFiles, hv_i;
  Htuple  hv_Selection;

  /* Local copy input parameter variables */
  Htuple   hv_Extensions_COPY_INP_TMP;


  /* Initialize control variables */
  create_tuple(&hv_ImageDirectoryIndex,0);
  create_tuple(&hv_ImageFilesTmp,0);
  create_tuple(&hv_CurrentImageDirectory,0);
  create_tuple(&hv_HalconImages,0);
  create_tuple(&hv_OS,0);
  create_tuple(&hv_Directories,0);
  create_tuple(&hv_Index,0);
  create_tuple(&hv_Length,0);
  create_tuple(&hv_NetworkDrive,0);
  create_tuple(&hv_Substring,0);
  create_tuple(&hv_FileExists,0);
  create_tuple(&hv_AllFiles,0);
  create_tuple(&hv_i,0);
  create_tuple(&hv_Selection,0);
  create_tuple(&(*hv_ImageFiles),0);
  copy_tuple(hv_Extensions,&hv_Extensions_COPY_INP_TMP);

  /****************************************************/
  /******************   Begin procedure   *************/
  /****************************************************/

  /*This procedure returns all files in a given directory*/
  /*with one of the suffixes specified in Extensions.*/
  /**/
  /*Input parameters:*/
  /*ImageDirectory: Directory or a tuple of directories with images.*/
  /*   If a directory is not found locally, the respective directory*/
  /*   is searched under %HALCONIMAGES%/ImageDirectory.*/
  /*   See the Installation Guide for further information*/
  /*   in case %HALCONIMAGES% is not set.*/
  /*Extensions: A string tuple containing the extensions to be found*/
  /*   e.g. ['png','tif',jpg'] or others*/
  /*If Extensions is set to 'default' or the empty string '',*/
  /*   all image suffixes supported by HALCON are used.*/
  /*Options: as in the operator list_files, except that the 'files'*/
  /*   option is always used. Note that the 'directories' option*/
  /*   has no effect but increases runtime, because only files are*/
  /*   returned.*/
  /**/
  /*Output parameter:*/
  /*ImageFiles: A tuple of all found image file names*/
  /**/
  /*========== if (Extensions == [] or Extensions == '' or Extensions == 'default') ==========*/
  create_tuple(&TTemp[SP++],0);
  T_tuple_equal(hv_Extensions_COPY_INP_TMP,TTemp[SP-1],&TTemp[SP]);
  destroy_tuple(TTemp[SP-1]);
  TTemp[SP-1]=TTemp[SP];
  create_tuple_s(&TTemp[SP++],"");
  T_tuple_equal(hv_Extensions_COPY_INP_TMP,TTemp[SP-1],&TTemp[SP]);
  destroy_tuple(TTemp[SP-1]);
  TTemp[SP-1]=TTemp[SP];
  T_tuple_or(TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
  destroy_tuple(TTemp[SP-2]);
  destroy_tuple(TTemp[SP-1]);
  TTemp[SP-2]=TTemp[SP];
  SP--;
  create_tuple_s(&TTemp[SP++],"default");
  T_tuple_equal(hv_Extensions_COPY_INP_TMP,TTemp[SP-1],&TTemp[SP]);
  destroy_tuple(TTemp[SP-1]);
  TTemp[SP-1]=TTemp[SP];
  T_tuple_or(TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
  destroy_tuple(TTemp[SP-2]);
  destroy_tuple(TTemp[SP-1]);
  TTemp[SP-2]=TTemp[SP];
  SP--;
  if(get_i(TTemp[SP-1],0))
  {
    /*Extensions := ['ima', 'tif', 'tiff', 'gif', 'bmp', 'jpg', 'jpeg', 'jp2', 'jxr', 'png', 'pcx', 'ras', 'xwd', 'pbm', 'pnm', 'pgm', 'ppm']*/
    create_tuple(&TTemp[SP++],17);
    set_s(TTemp[SP-1],"ima",0);
    set_s(TTemp[SP-1],"tif",1);
    set_s(TTemp[SP-1],"tiff",2);
    set_s(TTemp[SP-1],"gif",3);
    set_s(TTemp[SP-1],"bmp",4);
    set_s(TTemp[SP-1],"jpg",5);
    set_s(TTemp[SP-1],"jpeg",6);
    set_s(TTemp[SP-1],"jp2",7);
    set_s(TTemp[SP-1],"jxr",8);
    set_s(TTemp[SP-1],"png",9);
    set_s(TTemp[SP-1],"pcx",10);
    set_s(TTemp[SP-1],"ras",11);
    set_s(TTemp[SP-1],"xwd",12);
    set_s(TTemp[SP-1],"pbm",13);
    set_s(TTemp[SP-1],"pnm",14);
    set_s(TTemp[SP-1],"pgm",15);
    set_s(TTemp[SP-1],"ppm",16);
    destroy_tuple(hv_Extensions_COPY_INP_TMP);
    hv_Extensions_COPY_INP_TMP=TTemp[--SP];

    /**/
  }
  destroy_tuple(TTemp[--SP]);
  /*========== end if ==========*/
  /*ImageFiles := []*/
  create_tuple(&TTemp[SP++],0);
  destroy_tuple((*hv_ImageFiles));
  (*hv_ImageFiles)=TTemp[--SP];

  /*Loop through all given image directories.*/
  /*========== for ImageDirectoryIndex := 0 to |ImageDirectory| - 1 by 1 ==========*/
  T_tuple_length(hv_ImageDirectory,&TTemp[SP++]);
  create_tuple_i(&TTemp[SP++],1);
  T_tuple_sub(TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
  destroy_tuple(TTemp[SP-2]);
  destroy_tuple(TTemp[SP-1]);
  TTemp[SP-2]=TTemp[SP];
  SP--;
  create_tuple_i(&TTemp[SP++],1);
  create_tuple_i(&TTemp[SP++],0);
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
   destroy_tuple(hv_ImageDirectoryIndex);
   copy_tuple(TTemp[SP],&hv_ImageDirectoryIndex);
   destroy_tuple(TTemp[SP--]);
   destroy_tuple(TTemp[SP]);
   for(;;)
   {
   T_tuple_add(hv_ImageDirectoryIndex,TTemp[SP-1],&TTemp[SP]);
   destroy_tuple(hv_ImageDirectoryIndex);
   copy_tuple(TTemp[SP],&hv_ImageDirectoryIndex);
   destroy_tuple(TTemp[SP]);
   if(get_d(TTemp[SP-1],0)<0)
    T_tuple_less(hv_ImageDirectoryIndex,TTemp[SP-2],&TTemp[SP]);
   else
    T_tuple_greater(hv_ImageDirectoryIndex,TTemp[SP-2],&TTemp[SP]);
   if(get_i(TTemp[SP],0)) break;
   destroy_tuple(TTemp[SP]);
   /*========== for ==========*/

    /*ImageFilesTmp := []*/
    create_tuple(&TTemp[SP++],0);
    destroy_tuple(hv_ImageFilesTmp);
    hv_ImageFilesTmp=TTemp[--SP];

    /*CurrentImageDirectory := ImageDirectory[ImageDirectoryIndex]*/
    T_tuple_select(hv_ImageDirectory,hv_ImageDirectoryIndex,&TTemp[SP++]);
    destroy_tuple(hv_CurrentImageDirectory);
    hv_CurrentImageDirectory=TTemp[--SP];

    /*========== if (CurrentImageDirectory == '') ==========*/
    create_tuple_s(&TTemp[SP++],"");
    T_tuple_equal(hv_CurrentImageDirectory,TTemp[SP-1],&TTemp[SP]);
    destroy_tuple(TTemp[SP-1]);
    TTemp[SP-1]=TTemp[SP];
    if(get_i(TTemp[SP-1],0))
    {
      /*CurrentImageDirectory := '.'*/
      reuse_tuple_s(&hv_CurrentImageDirectory,".");
    }
    destroy_tuple(TTemp[--SP]);
    /*========== end if ==========*/
    /*get_system ('image_dir', HalconImages)*/
    create_tuple_s(&TTemp[SP++],"image_dir");
    destroy_tuple(hv_HalconImages);
    /***/T_get_system(TTemp[SP-1], &hv_HalconImages);
    destroy_tuple(TTemp[--SP]);

    /*get_system ('operating_system', OS)*/
    create_tuple_s(&TTemp[SP++],"operating_system");
    destroy_tuple(hv_OS);
    /***/T_get_system(TTemp[SP-1], &hv_OS);
    destroy_tuple(TTemp[--SP]);

    /*========== if (OS{0:2} == 'Win') ==========*/
    create_tuple_i(&TTemp[SP++],0);
    create_tuple_i(&TTemp[SP++],2);
    T_tuple_substr(hv_OS,TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
    destroy_tuple(TTemp[SP-2]);
    destroy_tuple(TTemp[SP-1]);
    TTemp[SP-2]=TTemp[SP];
    SP=SP-1;
    create_tuple_s(&TTemp[SP++],"Win");
    T_tuple_equal(TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
    destroy_tuple(TTemp[SP-2]);
    destroy_tuple(TTemp[SP-1]);
    TTemp[SP-2]=TTemp[SP];
    SP--;
    if(get_i(TTemp[SP-1],0))
    {
      /*HalconImages := split(HalconImages,';')*/
      copy_tuple(hv_HalconImages,&TTemp[SP++]);
      create_tuple_s(&TTemp[SP++],";");
      T_tuple_split(TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
      destroy_tuple(TTemp[SP-2]);
      destroy_tuple(TTemp[SP-1]);
      TTemp[SP-2]=TTemp[SP];
      SP--;
      destroy_tuple(hv_HalconImages);
      hv_HalconImages=TTemp[--SP];

    }
    else
    {
      /*HalconImages := split(HalconImages,':')*/
      copy_tuple(hv_HalconImages,&TTemp[SP++]);
      create_tuple_s(&TTemp[SP++],":");
      T_tuple_split(TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
      destroy_tuple(TTemp[SP-2]);
      destroy_tuple(TTemp[SP-1]);
      TTemp[SP-2]=TTemp[SP];
      SP--;
      destroy_tuple(hv_HalconImages);
      hv_HalconImages=TTemp[--SP];

    }
    destroy_tuple(TTemp[--SP]);
    /*========== end if ==========*/
    /*Directories := CurrentImageDirectory*/
    destroy_tuple(hv_Directories);
    copy_tuple(hv_CurrentImageDirectory,&hv_Directories);

    /*========== for Index := 0 to |HalconImages| - 1 by 1 ==========*/
    T_tuple_length(hv_HalconImages,&TTemp[SP++]);
    create_tuple_i(&TTemp[SP++],1);
    T_tuple_sub(TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
    destroy_tuple(TTemp[SP-2]);
    destroy_tuple(TTemp[SP-1]);
    TTemp[SP-2]=TTemp[SP];
    SP--;
    create_tuple_i(&TTemp[SP++],1);
    create_tuple_i(&TTemp[SP++],0);
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

      /*Directories := [Directories,HalconImages[Index] + '/' + CurrentImageDirectory]*/
      copy_tuple(hv_Directories,&TTemp[SP++]);
      T_tuple_select(hv_HalconImages,hv_Index,&TTemp[SP++]);
      create_tuple_s(&TTemp[SP++],"/");
      T_tuple_add(TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
      destroy_tuple(TTemp[SP-2]);
      destroy_tuple(TTemp[SP-1]);
      TTemp[SP-2]=TTemp[SP];
      SP--;
      T_tuple_add(TTemp[SP-1],hv_CurrentImageDirectory,&TTemp[SP]);
      destroy_tuple(TTemp[SP-1]);
      TTemp[SP-1]=TTemp[SP];
      T_tuple_concat(TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
      destroy_tuple(TTemp[SP-2]);
      destroy_tuple(TTemp[SP-1]);
      TTemp[SP-2]=TTemp[SP];
      SP--;
      destroy_tuple(hv_Directories);
      hv_Directories=TTemp[--SP];

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

    /*tuple_strlen (Directories, Length)*/
    destroy_tuple(hv_Length);
    /***/T_tuple_strlen(hv_Directories, &hv_Length);

    /*tuple_gen_const (|Length|, false, NetworkDrive)*/
    T_tuple_length(hv_Length,&TTemp[SP++]);
    create_tuple_i(&TTemp[SP++],0);
    destroy_tuple(hv_NetworkDrive);
    /***/T_tuple_gen_const(TTemp[SP-2], TTemp[SP-1], &hv_NetworkDrive);
    destroy_tuple(TTemp[--SP]);
    destroy_tuple(TTemp[--SP]);

    /*========== if (OS{0:2} == 'Win') ==========*/
    create_tuple_i(&TTemp[SP++],0);
    create_tuple_i(&TTemp[SP++],2);
    T_tuple_substr(hv_OS,TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
    destroy_tuple(TTemp[SP-2]);
    destroy_tuple(TTemp[SP-1]);
    TTemp[SP-2]=TTemp[SP];
    SP=SP-1;
    create_tuple_s(&TTemp[SP++],"Win");
    T_tuple_equal(TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
    destroy_tuple(TTemp[SP-2]);
    destroy_tuple(TTemp[SP-1]);
    TTemp[SP-2]=TTemp[SP];
    SP--;
    if(get_i(TTemp[SP-1],0))
    {
      /*========== for Index := 0 to |Length| - 1 by 1 ==========*/
      T_tuple_length(hv_Length,&TTemp[SP++]);
      create_tuple_i(&TTemp[SP++],1);
      T_tuple_sub(TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
      destroy_tuple(TTemp[SP-2]);
      destroy_tuple(TTemp[SP-1]);
      TTemp[SP-2]=TTemp[SP];
      SP--;
      create_tuple_i(&TTemp[SP++],1);
      create_tuple_i(&TTemp[SP++],0);
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

        /*========== if (strlen(Directories[Index]) > 1) ==========*/
        T_tuple_select(hv_Directories,hv_Index,&TTemp[SP++]);
        T_tuple_strlen(TTemp[SP-1],&TTemp[SP]);
        destroy_tuple(TTemp[SP-1]);
        TTemp[SP-1]=TTemp[SP];
        create_tuple_i(&TTemp[SP++],1);
        T_tuple_greater(TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
        destroy_tuple(TTemp[SP-2]);
        destroy_tuple(TTemp[SP-1]);
        TTemp[SP-2]=TTemp[SP];
        SP--;
        if(get_i(TTemp[SP-1],0))
        {
          /*tuple_str_first_n (Directories[Index], 1, Substring)*/
          T_tuple_select(hv_Directories,hv_Index,&TTemp[SP++]);
          create_tuple_i(&TTemp[SP++],1);
          destroy_tuple(hv_Substring);
          /***/T_tuple_str_first_n(TTemp[SP-2], TTemp[SP-1], &hv_Substring);
          destroy_tuple(TTemp[--SP]);
          destroy_tuple(TTemp[--SP]);

          /*========== if (Substring == '//' or Substring == '\\\\') ==========*/
          create_tuple_s(&TTemp[SP++],"//");
          T_tuple_equal(hv_Substring,TTemp[SP-1],&TTemp[SP]);
          destroy_tuple(TTemp[SP-1]);
          TTemp[SP-1]=TTemp[SP];
          create_tuple_s(&TTemp[SP++],"\\\\");
          T_tuple_equal(hv_Substring,TTemp[SP-1],&TTemp[SP]);
          destroy_tuple(TTemp[SP-1]);
          TTemp[SP-1]=TTemp[SP];
          T_tuple_or(TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
          destroy_tuple(TTemp[SP-2]);
          destroy_tuple(TTemp[SP-1]);
          TTemp[SP-2]=TTemp[SP];
          SP--;
          if(get_i(TTemp[SP-1],0))
          {
            /*NetworkDrive[Index] := true*/
            create_tuple_i(&TTemp[SP++],1);
            replace_elements(&hv_NetworkDrive,&hv_Index,&TTemp[SP-1]);
            destroy_tuple(TTemp[--SP]);
          }
          destroy_tuple(TTemp[--SP]);
          /*========== end if ==========*/
        }
        destroy_tuple(TTemp[--SP]);
        /*========== end if ==========*/
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

    }
    destroy_tuple(TTemp[--SP]);
    /*========== end if ==========*/
    /*ImageFilesTmp := []*/
    create_tuple(&TTemp[SP++],0);
    destroy_tuple(hv_ImageFilesTmp);
    hv_ImageFilesTmp=TTemp[--SP];

    /*========== for Index := 0 to |Directories| - 1 by 1 ==========*/
    T_tuple_length(hv_Directories,&TTemp[SP++]);
    create_tuple_i(&TTemp[SP++],1);
    T_tuple_sub(TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
    destroy_tuple(TTemp[SP-2]);
    destroy_tuple(TTemp[SP-1]);
    TTemp[SP-2]=TTemp[SP];
    SP--;
    create_tuple_i(&TTemp[SP++],1);
    create_tuple_i(&TTemp[SP++],0);
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

      /*file_exists (Directories[Index], FileExists)*/
      T_tuple_select(hv_Directories,hv_Index,&TTemp[SP++]);
      destroy_tuple(hv_FileExists);
      /***/T_file_exists(TTemp[SP-1], &hv_FileExists);
      destroy_tuple(TTemp[--SP]);

      /*========== if (FileExists) ==========*/
      copy_tuple(hv_FileExists,&TTemp[SP++]);
      if(get_i(TTemp[SP-1],0))
      {
        /*list_files (Directories[Index], ['files',Options], AllFiles)*/
        T_tuple_select(hv_Directories,hv_Index,&TTemp[SP++]);
        create_tuple_s(&TTemp[SP++],"files");
        T_tuple_concat(TTemp[SP-1],hv_Options,&TTemp[SP]);
        destroy_tuple(TTemp[SP-1]);
        TTemp[SP-1]=TTemp[SP];
        destroy_tuple(hv_AllFiles);
        /***/T_list_files(TTemp[SP-2], TTemp[SP-1], &hv_AllFiles);
        destroy_tuple(TTemp[--SP]);
        destroy_tuple(TTemp[--SP]);

        /*ImageFilesTmp := []*/
        create_tuple(&TTemp[SP++],0);
        destroy_tuple(hv_ImageFilesTmp);
        hv_ImageFilesTmp=TTemp[--SP];

        /*========== for i := 0 to |Extensions| - 1 by 1 ==========*/
        T_tuple_length(hv_Extensions_COPY_INP_TMP,&TTemp[SP++]);
        create_tuple_i(&TTemp[SP++],1);
        T_tuple_sub(TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
        destroy_tuple(TTemp[SP-2]);
        destroy_tuple(TTemp[SP-1]);
        TTemp[SP-2]=TTemp[SP];
        SP--;
        create_tuple_i(&TTemp[SP++],1);
        create_tuple_i(&TTemp[SP++],0);
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
         destroy_tuple(hv_i);
         copy_tuple(TTemp[SP],&hv_i);
         destroy_tuple(TTemp[SP--]);
         destroy_tuple(TTemp[SP]);
         for(;;)
         {
         T_tuple_add(hv_i,TTemp[SP-1],&TTemp[SP]);
         destroy_tuple(hv_i);
         copy_tuple(TTemp[SP],&hv_i);
         destroy_tuple(TTemp[SP]);
         if(get_d(TTemp[SP-1],0)<0)
          T_tuple_less(hv_i,TTemp[SP-2],&TTemp[SP]);
         else
          T_tuple_greater(hv_i,TTemp[SP-2],&TTemp[SP]);
         if(get_i(TTemp[SP],0)) break;
         destroy_tuple(TTemp[SP]);
         /*========== for ==========*/

          /*tuple_regexp_select (AllFiles, ['.*' + Extensions[i] + '$','ignore_case'], Selection)*/
          create_tuple_s(&TTemp[SP++],".*");
          T_tuple_select(hv_Extensions_COPY_INP_TMP,hv_i,&TTemp[SP++]);
          T_tuple_add(TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
          destroy_tuple(TTemp[SP-2]);
          destroy_tuple(TTemp[SP-1]);
          TTemp[SP-2]=TTemp[SP];
          SP--;
          create_tuple_s(&TTemp[SP++],"$");
          T_tuple_add(TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
          destroy_tuple(TTemp[SP-2]);
          destroy_tuple(TTemp[SP-1]);
          TTemp[SP-2]=TTemp[SP];
          SP--;
          create_tuple_s(&TTemp[SP++],"ignore_case");
          T_tuple_concat(TTemp[SP-2],TTemp[SP-1],&TTemp[SP]);
          destroy_tuple(TTemp[SP-2]);
          destroy_tuple(TTemp[SP-1]);
          TTemp[SP-2]=TTemp[SP];
          SP--;
          destroy_tuple(hv_Selection);
          /***/T_tuple_regexp_select(hv_AllFiles, TTemp[SP-1], &hv_Selection);
          destroy_tuple(TTemp[--SP]);

          /*ImageFilesTmp := [ImageFilesTmp,Selection]*/
          copy_tuple(hv_ImageFilesTmp,&TTemp[SP++]);
          T_tuple_concat(TTemp[SP-1],hv_Selection,&TTemp[SP]);
          destroy_tuple(TTemp[SP-1]);
          TTemp[SP-1]=TTemp[SP];
          destroy_tuple(hv_ImageFilesTmp);
          hv_ImageFilesTmp=TTemp[--SP];

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

        /*tuple_regexp_replace (ImageFilesTmp, ['\\\\', 'replace_all'], '/', ImageFilesTmp)*/
        copy_tuple(hv_ImageFilesTmp,&TTemp[SP++]);
        create_tuple(&TTemp[SP++],2);
        set_s(TTemp[SP-1],"\\\\",0);
        set_s(TTemp[SP-1],"replace_all",1);
        create_tuple_s(&TTemp[SP++],"/");
        destroy_tuple(hv_ImageFilesTmp);
        /***/T_tuple_regexp_replace(TTemp[SP-3], TTemp[SP-2], TTemp[SP-1], &hv_ImageFilesTmp);
        destroy_tuple(TTemp[--SP]);
        destroy_tuple(TTemp[--SP]);
        destroy_tuple(TTemp[--SP]);

        /*========== if (NetworkDrive[Index]) ==========*/
        T_tuple_select(hv_NetworkDrive,hv_Index,&TTemp[SP++]);
        if(get_i(TTemp[SP-1],0))
        {
          /*tuple_regexp_replace (ImageFilesTmp, ['//', 'replace_all'], '/', ImageFilesTmp)*/
          copy_tuple(hv_ImageFilesTmp,&TTemp[SP++]);
          create_tuple(&TTemp[SP++],2);
          set_s(TTemp[SP-1],"//",0);
          set_s(TTemp[SP-1],"replace_all",1);
          create_tuple_s(&TTemp[SP++],"/");
          destroy_tuple(hv_ImageFilesTmp);
          /***/T_tuple_regexp_replace(TTemp[SP-3], TTemp[SP-2], TTemp[SP-1], &hv_ImageFilesTmp);
          destroy_tuple(TTemp[--SP]);
          destroy_tuple(TTemp[--SP]);
          destroy_tuple(TTemp[--SP]);

          /*ImageFilesTmp := '/' + ImageFilesTmp*/
          create_tuple_s(&TTemp[SP++],"/");
          T_tuple_add(TTemp[SP-1],hv_ImageFilesTmp,&TTemp[SP]);
          destroy_tuple(hv_ImageFilesTmp);
          hv_ImageFilesTmp=TTemp[SP];
          destroy_tuple(TTemp[--SP]);

        }
        else
        {
          /*tuple_regexp_replace (ImageFilesTmp, ['//', 'replace_all'], '/', ImageFilesTmp)*/
          copy_tuple(hv_ImageFilesTmp,&TTemp[SP++]);
          create_tuple(&TTemp[SP++],2);
          set_s(TTemp[SP-1],"//",0);
          set_s(TTemp[SP-1],"replace_all",1);
          create_tuple_s(&TTemp[SP++],"/");
          destroy_tuple(hv_ImageFilesTmp);
          /***/T_tuple_regexp_replace(TTemp[SP-3], TTemp[SP-2], TTemp[SP-1], &hv_ImageFilesTmp);
          destroy_tuple(TTemp[--SP]);
          destroy_tuple(TTemp[--SP]);
          destroy_tuple(TTemp[--SP]);

        }
        destroy_tuple(TTemp[--SP]);
        /*========== end if ==========*/
        /*break*/
        destroy_tuple(TTemp[--SP]);
        create_tuple(&TTemp[SP],0);
        break;
      }
      destroy_tuple(TTemp[--SP]);
      /*========== end if ==========*/
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

    /*Concatenate the output image paths.*/
    /*ImageFiles := [ImageFiles,ImageFilesTmp]*/
    copy_tuple((*hv_ImageFiles),&TTemp[SP++]);
    T_tuple_concat(TTemp[SP-1],hv_ImageFilesTmp,&TTemp[SP]);
    destroy_tuple(TTemp[SP-1]);
    TTemp[SP-1]=TTemp[SP];
    destroy_tuple((*hv_ImageFiles));
    (*hv_ImageFiles)=TTemp[--SP];

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

  /*========== return ==========*/

  /* Clear temporary tuple stack */
  while (SP > 0)
    destroy_tuple(TTemp[--SP]);
  /* Clear temporary tuple vectors stack*/
  while (SPTV > 0)
    V_destroy_vector(TVTemp[--SPTV]);
  /* Clear local control variables */
  destroy_tuple(hv_ImageDirectoryIndex);
  destroy_tuple(hv_ImageFilesTmp);
  destroy_tuple(hv_CurrentImageDirectory);
  destroy_tuple(hv_HalconImages);
  destroy_tuple(hv_OS);
  destroy_tuple(hv_Directories);
  destroy_tuple(hv_Index);
  destroy_tuple(hv_Length);
  destroy_tuple(hv_NetworkDrive);
  destroy_tuple(hv_Substring);
  destroy_tuple(hv_FileExists);
  destroy_tuple(hv_AllFiles);
  destroy_tuple(hv_i);
  destroy_tuple(hv_Selection);
  destroy_tuple(hv_Extensions_COPY_INP_TMP);

  return;

  /****************************************************/
  /******************     End procedure   *************/
  /****************************************************/

}
