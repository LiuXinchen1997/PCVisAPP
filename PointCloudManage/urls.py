from django.urls import path, re_path
from django.views.generic import TemplateView

from PointCloudManage import views


app_name = 'PointCloudManage'


urlpatterns = [
    path('', views.index, name='index'),
    path('index/', views.index, name='index'),
    path('pcupsample-display-page/', views.pcupsample_display_page, name='pcupsample-display-page'),
    path('pcupsample-vis-page/<str:pc_name>/', views.pcupsample_vis_page, name='pcupsample-vis-page'),

    path('sceneupsample-display-page/', views.sceneupsample_display_page, name='sceneupsample-display-page'),
    path('sceneupsample-vis-page/', views.sceneupsample_vis_page, name='sceneupsample-vis-page'),

    path('segdataset-display-page/', views.segdataset_display_page, name='segdataset-display-page'),
    path('segdataset-vis-page/', views.segdataset_vis_page, name='segdataset-vis-page'),

    path('sceneseg-display-page/', views.sceneseg_display_page, name='sceneseg-display-page'),
    path('sceneseg-vis-page/', views.sceneseg_vis_page, name='sceneseg-vis-page'),

    path('mypcbase-display-page/', views.mypcbase_display_page, name='mypcbase-display-page'),
    path('mypcbase-folder-display-page/', views.mypcbase_folder_display_page, name='mypcbase-folder-display-page'),
    path('mypcbase-upload-page/', views.mypcbase_upload_page, name='mypcbase-upload-page'),
    path('mypcbase-upload/', views.mypcbase_upload, name='mypcbase-upload'),
    path('mypcbase-delete/', views.mypcbase_delete, name='mypcbase-delete'),
    path('mypcbase-folder-delete-all/', views.mypcbase_folder_delete_all, name='mypcbase-folder-delete-all'),
    path('mypcbase-vis-page/<str:folder_name>/<str:pc_name>/', views.mypcbase_vis_page, name='mypcbase-vis-page'),
    path('mypcbase-save-ply-with-rgb/', views.mypcbase_save_ply_with_rgb, name='mypcbase-save-ply-with-rgb'),
    path('mypcbase-get-texture-path/', views.mypcbase_get_texture_path, name='mypcbase-get-texture-path'),
    path('mypcbase-upsample/', views.mypcbase_upsample, name='mypcbase-upsample'),
    path('mypcbase-vis-upsample-page/<str:folder_name>/<str:pc_name>/', views.mypcbase_vis_upsample_page, name='mypcbase-vis-upsample-page'),
    path('mypcbase-zip-all/', views.mypcbase_zip_all, name="mypcbase-zip-all"),
    path('mypcbase-folder-zip-all/', views.mypcbase_folder_zip_all, name="mypcbase-folder-zip-all"),
    path('mypcbase-clear-temp-dir/', views.mypcbase_clear_temp_dir, name='mypcbase-clear-temp-dir'),
    path('mypcbase-get-progress-val/', views.mypcbase_get_progress_val, name='mypcbase-get-progress-val'),
    path('mypcbase-create-folder/', views.mypcbase_create_folder, name='mypcbase-create-folder'),
    path('mypcbase-rename-folder/', views.mypcbase_rename_folder, name='mypcbase-rename-folder'),
    path('mypcbase-folder-move-pc/', views.mypcbase_folder_move_pc, name='mypcbase-folder-move-pc'),

    path('recyclebin-display-page/', views.recyclebin_display_page, name='recyclebin-display-page'),
    path('recyclebin-folder-display-page/', views.recyclebin_folder_display_page, name='recyclebin-folder-display-page'),
    path('recyclebin-delete/', views.recyclebin_delete, name='recyclebin-delete'),
    path('recyclebin-folder-delete-all/', views.recyclebin_folder_delete_all, name='recyclebin-folder-delete-all'),
    path('recyclebin-delete-all/', views.recyclebin_delete_all, name='recyclebin-delete-all'),
    path('recyclebin-restore/', views.recyclebin_restore, name='recyclebin-restore'),
    path('recyclebin-folder-restore-all/', views.recyclebin_folder_restore_all, name='recyclebin-folder-restore-all'),
    path('recyclebin-restore-all/', views.recyclebin_restore_all, name='recyclebin-restore-all'),
    path('test/', views.test, name='test'),
    re_path(r'[a-zA-Z0-9]', TemplateView.as_view(template_name='404.html')),
]

handler404 = TemplateView.as_view(template_name='404.html')
