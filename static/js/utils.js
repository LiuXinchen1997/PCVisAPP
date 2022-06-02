let CHINESE2ENGLISH = {
    '纯白': 'White',
    '浅灰': 'Gainsboro',
    '亚麻色': 'Linen',
    '柠檬黄': 'LemonChiffon',
    '蜜瓜绿': 'Honeydew',
    '淡绿': 'PaleGreen1',
    '淡蓝': 'LightBlue',
    '橙红': 'Salmon',
    '纯黑': 'Black',
}

let COLORNAME2RGB = {
    'White': 0xFFFFFF,
    'Gainsboro': 0xDCDCDC,
    'Linen': 0xFAF0E6,
    'LemonChiffon': 0xFFFACD,
    'Honeydew': 0xF0FFF0,
    'PaleGreen1': 0x9AFF9A,
    'LightBlue': 0xADD8E6,
    'Salmon': 0xFA8072,
    'Black': 0x000000,
};


function vis(obj_file_path, doc_canvas_id, texture_file_path=null, opts={}) {
    function _get_width () {
        let canvas = $('#' + doc_canvas_id);
        let width = parseInt(canvas.css('width').slice(0, -2));
        
        if (opts['width'] !== undefined) {
            width = opts['width'];
        } else if (opts['width_ratio'] !== undefined) {
            width = window.innerWidth / opts['width_ratio'];
        }
        return width;
    }

    function _get_height () {
        let canvas = $('#' + doc_canvas_id);
        let height = parseInt(canvas.css('height').slice(0, -2));
        
        if (opts['height'] !== undefined) {
            height = opts['height'];
        } else if (opts['height_ratio'] !== undefined) {
            height = window.innerHeight / opts['height_ratio'];
        }
        return height;
    }

    let width = _get_width();
    let height = _get_height();

    rotated = true;
    if (opts['rotated'] !== undefined) {
        rotated = opts['rotated'];
    }
    
    let ambient_light_color = 0xFFFFFF;
    if (opts['ambient_light_color'] !== undefined) { ambient_light_color = opts['ambient_light_color']; }
    let direct_light_color = 0xFFFFFF;
    if (opts['direct_light_color'] !== undefined) { direct_light_color = opts['direct_light_color']; }
    let direct_light_intensity = 0.5;
    if (opts['direct_light_intensity'] !== undefined) { direct_light_intensity = opts['direct_light_intensity']; }
    let rotate_speed = 0.01;
    if (opts['rotate_speed'] !== undefined) { rotate_speed = opts['rotate_speed']; }
    let rotate_axis = 'y';
    if (opts['rotate_axis'] !== undefined) { rotate_axis = opts['rotate_axis']; }
    let canvas_color = 0xFFFFFF;
    if (opts['canvas_color'] !== undefined) { canvas_color = COLORNAME2RGB[CHINESE2ENGLISH[opts['canvas_color']]]; }

    let requestAnimationFrame = (function () {
        return window.requestAnimationFrame ||
            window.webkitRequestAnimationFrame ||
            window.mozRequestAnimationFrame ||
            window.oRequestAnimationFrame ||
            window.msRequestAnimationFrame ||
            function (callback) {
                window.setTimeout(callback, 1000 / 60);
            };
    })();

    let info = document.getElementById(doc_canvas_id);
    let scene; //用来盛放模型的场景
    let camera; //呈现模型的相机
    let renderer; //渲染模型的渲染器
    let control; //操作模型的控制器
    let objLoader; //加载obj模型的加载器
    let group = new THREE.Group();
    
    //场景内模型渲染准备
    function prepareRender() {
        scene = new THREE.Scene();
        scene.background = new THREE.Color(canvas_color);
        camera = new THREE.PerspectiveCamera(70, width / height, 1, 10000000000);
        renderer = new THREE.WebGLRenderer();
        renderer.autoClear = false;

        //初始化相机位置。
        camera.position.x = 150;
        camera.position.y = 150;
        camera.position.z = 150;
        renderer.setSize(width, height);

        //将渲染画布放到dom元素中，即前面声明的div。
        info.appendChild(renderer.domElement);

        //声明控制器，传入相机和被控制的dom节点。
        control = new THREE.OrbitControls(camera, renderer.domElement.parentNode);

        //控制器在控制元素时围绕的中心位置。
        control.target = new THREE.Vector3(0, 0, 0);

        //相机的朝向
        camera.aspect = window.innerWidth / window.innerHeight;
    }

    //向场景内添加obj模型
    function insertObj() {
        //初始化OBJLoader加载器。
        objLoader = new THREE.OBJLoader();
        
        //创建模型的纹理（贴图）加载器。
        let texture = null;
        if (texture_file_path != null) {
            let textureLoader = new THREE.TextureLoader();
            texture = textureLoader.load(texture_file_path);
        }
        
        //加载模型
        objLoader.load(obj_file_path, function (object) {
            object.traverse(function (child) {
                //将加载到的纹理给模型的材质
                if (texture != null && child instanceof THREE.Mesh) {
                    child.material.map = texture;
                }
            });

            let scale_ratio = opts['scale_ratio'];
            object.scale.set(scale_ratio, scale_ratio, scale_ratio);

            // 将模型的位置始终定位在中心点(0,0,0)
            // 这一步的操作是为了配合模型控制器的效果，前面的模型控制器就是中心点就是设置在(0,0,0)位置的。
            // 用户在用鼠标旋转模型时，好像在围绕着模型的中心旋转。
            var box = new THREE.Box3().setFromObject(object);
            var center = box.getCenter();//用一个Box获取到模型的当前位置。
            object.position.set(-center.x, -center.y, -center.z);//将模型移回原点。

            group.add(object);
            scene.add(group);
        },
        //进度回调函数
        function (xhr) {
            // console.log( ( xhr.loaded / xhr.total * 100 ) + '% loaded' );
            if (opts['doc_progress_id'] !== undefined && opts['doc_progress_bar_id'] !== undefined) {
                let progress = $('#' + opts['doc_progress_id']);
                // progress.width('' + _get_width() + 'px');
                let progress_bar = $('#' + opts['doc_progress_bar_id']);
                // progress_bar.width('' + _get_width() + 'px');
                let progress_val = parseInt(xhr.loaded / xhr.total * 100);
                progress_bar.attr('aria-valuenow', progress_val);
                progress_bar.attr('style', 'width:'+progress_val+'%');
                progress_bar.html(''+progress_val+'%');

                if (progress_val === 100) {
                    progress_bar.addClass('progress-bar-success'); 
                    progress_bar.html('加载完成');
                    if (opts['progress_hide']) {
                        setTimeout(function () {
                            progress.hide();
                        }, 2000);
                    } 
                }
            }
        },
        // called when loading has errors
        function (error) {
            // console.log( 'An error happened' );
            openErrMsgBox('<strong>抱歉！</strong> 模型加载失败：' + error);
        });
    }

    //场景内添加灯
    function insertOther() {
        //环境光
        // let light = new THREE.AmbientLight( 0x404040 ); // soft white light
        let light = new THREE.AmbientLight(ambient_light_color);
        scene.add(light);

        //方向光
        let directionalLight = new THREE.DirectionalLight(direct_light_color, direct_light_intensity);
        scene.add(directionalLight);
    }
    
    function render() {
        width = _get_width();
        height = _get_height();
        renderer.setSize(width, height);
        renderer.render(scene, camera);
    }
    
    let animate_id = requestAnimationFrame(animate);
    function animate() {
        if ($('#' + doc_canvas_id).length === 0) { // stop animate
            cancelAnimationFrame(animate_id);
            return;
        }

        if (rotated) {
            if ('x' === rotate_axis) {
                group.rotation.x -= rotate_speed;
            } else if ('y' === rotate_axis) {
                group.rotation.y -= rotate_speed;
            } else if ('z' === rotate_axis) {
                group.rotation.z -= rotate_speed;
            } else if ('-x' === rotate_axis) {
                group.rotation.x += rotate_speed;
            } else if ('-y' === rotate_axis) {
                group.rotation.y += rotate_speed;
            } else {
                group.rotation.z += rotate_speed;
            }
        }
        control.update();
        requestAnimationFrame(animate);
        render();
    }

    function init() {
        prepareRender();
        insertObj();
        insertOther();
        animate();
    }

    //调用代码
    init();
}


function ply_vis(ply_file_path, doc_canvas_id, opts={}) {
    function _get_width () {
        let canvas = $('#' + doc_canvas_id);
        let width = parseInt(canvas.css('width').slice(0, -2));
        
        if (opts['width'] !== undefined) {
            width = opts['width'];
        } else if (opts['width_ratio'] !== undefined) {
            width = window.innerWidth / opts['width_ratio'];
        }
        return width;
    }

    function _get_height () {
        let canvas = $('#' + doc_canvas_id);
        let height = parseInt(canvas.css('height').slice(0, -2));
        
        if (opts['height'] !== undefined) {
            height = opts['height'];
        } else if (opts['height_ratio'] !== undefined) {
            height = window.innerHeight / opts['height_ratio'];
        }
        return height;
    }

    let width = _get_width();
    let height = _get_height();

    rotated = true;
    if (opts['rotated'] !== undefined) {
        rotated = opts['rotated'];
    }

    let point_size = 0.005;
    if (opts['point_size'] !== undefined) { point_size = opts['point_size']; }
    let rotate_speed = 0.01;
    if (opts['rotate_speed'] !== undefined) { rotate_speed = opts['rotate_speed']; }
    let rotate_axis = 'y';
    if (opts['rotate_axis'] !== undefined) { rotate_axis = opts['rotate_axis']; }
    let canvas_color = 0xFFFFFF;
    if (opts['canvas_color'] !== undefined) { canvas_color = COLORNAME2RGB[CHINESE2ENGLISH[opts['canvas_color']]]; }

    let requestAnimationFrame = (function () {
        return window.requestAnimationFrame ||
            window.webkitRequestAnimationFrame ||
            window.mozRequestAnimationFrame ||
            window.oRequestAnimationFrame ||
            window.msRequestAnimationFrame ||
            function (callback) {
                window.setTimeout(callback, 1000 / 60);
            };
    })();

    let info = document.getElementById(doc_canvas_id);
    let scene; //用来盛放模型的场景
    let camera; //呈现模型的相机
    let renderer; //渲染模型的渲染器
    let control; //操作模型的控制器
    let objLoader; //加载obj模型的加载器
    let group = new THREE.Group();
    
    //场景内模型渲染准备
    function prepareRender() {
        scene = new THREE.Scene();
        scene.background = new THREE.Color(canvas_color);
        camera = new THREE.PerspectiveCamera(70, width / height, 1, 10000000000);
        renderer = new THREE.WebGLRenderer();
        renderer.autoClear = false;

        //初始化相机位置。
        camera.position.x = 150;
        camera.position.y = 150;
        camera.position.z = 150;
        renderer.setSize(width, height);

        //将渲染画布放到dom元素中，即前面声明的div。
        info.appendChild(renderer.domElement);

        //声明控制器，传入相机和被控制的dom节点。
        control = new THREE.OrbitControls(camera, renderer.domElement.parentNode);

        //控制器在控制元素时围绕的中心位置。
        control.target = new THREE.Vector3(0, 0, 0);

        //相机的朝向
        camera.aspect = window.innerWidth / window.innerHeight;
    }

    //向场景内添加obj模型
    function insertObj() {
        //辅助工具
        let loader = new THREE.PLYLoader();
        loader.load(ply_file_path, function (geometry) {
            let scale_ratio = opts['scale_ratio'];
            //更新顶点的法向量
            geometry.computeVertexNormals();

            //创建纹理，并将模型添加到场景道中
            let material = new THREE.PointsMaterial({size: point_size})
            material.vertexColors = true;
            var mesh = new THREE.Points( geometry, material );

            mesh.scale.set(scale_ratio, scale_ratio, scale_ratio);
            //mesh.rotation.y = Math.PI;
            //mesh.scale.set(0.05, 0.05, 0.05);
            group.add(mesh);
            scene.add(group);
        }, 
        function (xhr) {
            // console.log( ( xhr.loaded / xhr.total * 100 ) + '% loaded' );
            if (opts['doc_progress_id'] !== undefined && opts['doc_progress_bar_id'] !== undefined) {
                let progress = $('#' + opts['doc_progress_id']);
                let progress_bar = $('#' + opts['doc_progress_bar_id']);
                
                let progress_val = parseInt(xhr.loaded / xhr.total * 100);
                progress_bar.attr('aria-valuenow', progress_val);
                progress_bar.attr('style', 'width:'+progress_val+'%');
                progress_bar.html(''+progress_val+'%');

                if (progress_val === 100) {
                    progress_bar.addClass('progress-bar-success'); 
                    progress_bar.html('加载完成');
                    if (opts['progress_hide']) {
                        setTimeout(function () {
                            progress.hide();
                        }, 2000);
                    } 
                }
            }
        },
        function (error) {
            //
        });
    }

    //场景内添加灯
    function insertOther() {
        //环境光
        // let light = new THREE.AmbientLight( 0x404040 ); // soft white light
        let light = new THREE.AmbientLight(0xffffff);
        scene.add(light);

        //方向光
        let directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        scene.add(directionalLight);
    }
    
    function render() {
        width = _get_width();
        height = _get_height();
        renderer.setSize(width, height);
        renderer.render(scene, camera);
    }
    
    let animate_id = requestAnimationFrame(animate);
    function animate() {
        if ($('#' + doc_canvas_id).length === 0) { // stop animate
            cancelAnimationFrame(animate_id);
            return;
        }

        if (rotated) { 
            if ('x' === rotate_axis) {
                group.rotation.x -= rotate_speed;
            } else if ('y' === rotate_axis) {
                group.rotation.y -= rotate_speed;
            } else if ('z' === rotate_axis) {
                group.rotation.z -= rotate_speed;
            } else if ('-x' === rotate_axis) {
                group.rotation.x += rotate_speed;
            } else if ('-y' === rotate_axis) {
                group.rotation.y += rotate_speed;
            } else {
                group.rotation.z += rotate_speed;
            }
        }
        control.update();
        requestAnimationFrame(animate);
        render();
    }

    function init() {
        prepareRender();
        insertObj();
        insertOther();
        animate();
    }

    init();
}


function openSuccMsgBox (msg, duration=5000) {
    $('#succ-msg-box-body').html(msg)
    $('#succ-msg-box').modal('show');
    setTimeout(function() {
        $('#succ-msg-box').modal('hide');
    }, duration);
}

function openErrMsgBox (msg, duration=5000) {
    $('#err-msg-box-body').html(msg)
    $('#err-msg-box').modal('show');
    setTimeout(function() {
        $('#err-msg-box').modal('hide');
    }, duration);
}

function openLoadingBox (msg, progress_show=false) {
    if (!progress_show) {
        $('#progress-loading-box').hide();
    }
    $('#loading-body').html(msg)
    $('#loading').modal('show');
}

function closeLoadingBox() {
    $('#loading').modal('hide');
}

function objClone(obj) {
    return JSON.parse(JSON.stringify(obj));
}

function RGB2Hex(red, green, blue) {
    let hex = 0;
    hex += red;
    hex <<= 8;
    hex += green;
    hex <<= 8;
    hex += blue;
    return hex;
}
