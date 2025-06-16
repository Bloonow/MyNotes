> **《Android开发艺术探索》**
>
> ***Android Programming: The Big Nerd Ranch Guide, Third Edition***
>
> ***[美]Bill Phillips，Chris Stewart，Kristin Marsicano，王明发 译，2017.7***

[TOC]

# 一、基础知识

Android的main函数在`ActivityThread`类中。

## （一）概述

MCV设计模式（Model-View-Controller），模型对象、视图对象、控制器对象。

XML文件设计：

```xml
<some.widget>
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    <some.widget/>
</some.widget>
```

组件有`LinearLayout`、`TextView`、`Button`、`CheckBox`、`ImageView`、`ImageButton`等。组件包含组件属性、资源id等。引用组件可以`findViewById(R.XXX)`，如获得应用名：`String appName = getResources().getString(R.string.app_name);`。

资源文件有字符串资源`strins.xml`等。

activity的生命周期：`onCreate()`、`onStart()`、`onResume()`、`onPause()`、`onStop()`、`onDestory()`。设备配置改变（如旋转屏幕）后导致资源变化造成activity被干掉然后重启，重写activity的方法以应对数据保存问题：`@Override protected void onSaveInstanceState(Bundle outState);`，重启时可以在onCreate中恢复数据：`if(savedInstanceState != null) { xxx }`。

设置断点。

Android SDK版本（API）：SDK最低版本、SDK目标版本、SDK编译版本。

图形布局工具`ConstraintLayout`，布局属性如px、dp、sp、pt、mm、in、style、text、size、margin、padding等。

此外还有MVVM（Model-View-ViewModel）架构、MVP（Model-View-Presenter）架构。

## （二）基本逻辑

### 1. 设置监听器

```java
mButton = findViewById(R.id.id_ma_button);
mButton.setOnClickListener(new View.OnClickListener() {
    @Override
    public void onClick(View v) {
        Toast.makeText(MainActivity.this, "Hello", Toast.LENGTH_SHORT).show();    // 创建Toast提示消息
    }
});
```

另外，长按事件的监听器是`View.OnLongClickListener`。

### 2. 组件EditText和CheckBox

```java
mEditText = findViewById(R.id.id_ma_edit_text);
mEditText.addTextChangedListener(new TextWatcher() {
    @Override
    public void beforeTextChanged(CharSequence s, int start, int count, int after) { }
    @Override
    public void onTextChanged(CharSequence s, int start, int before, int count) { }
    @Override
    public void afterTextChanged(Editable s) { }
});

mCheckBox = findViewById(R.id.id_ma_check_box);
mCheckBox.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
    @Override
    public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) { }
});
```

### 3. 用new Intent启动一个新activity

```java
// MainActivity.java
Intent intent = new Intent(MainActivity.this, SecondActivity.class);
intent.putExtra("key", "Hello");    // 可对Intent添加额外附加信息
startActivity(intent);    // 启动新activity
// 或者
startActivityForResult(intent, Request_Code);    // 可以通过intent从所启动的新activity中获取结果（如果有）
```

在新的activity中进行相应操作：

```java
// SecondActivity.java
String extraValue = getIntent().getStringExtra("key");  // 可以获得附加信息

Intent data = new Intent();    // 通过一个结果Intent向启动者activity返回结果
data.putExtra("result", "Yeah");
setResult(Result_Code, data);
```

在原来的activity中可以获得新activity返回的结果：

```java
// MainActivity.java
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (data == null || requestCode != Request_Code) return;
        switch (resultCode) {
            case SecondActivity.Result_Code:
                String resultValue = data.getStringExtra("result");
                mButton.setText(resultValue);
                break;
            default:
                super.onActivityResult(requestCode, resultCode, data);
        }
    }
```

可以在AndroidManifest中设置新Activity的parentActivityName属性，实现层级导航：

```java
<activity android:name=".SecondActivity" 
        android:parentActivityName=".MainActivity"/>
```

若当前显示在子activity中，按后退键向上导航回到父activity中，会重建父级activity，可以在子activity中使用`getParentActivityIntent()`方法获得Intent，用附带extra的Intent重建父级Activity。

### 4. AndroidManifest

```xml
<manifest>
    <application>
        <!-- xxx -->
    </application>
    
    <uses-feature
        android:name="android.hardware.Camera"
        android:required="false" />
    <uses-permission android:name="android.permission.READ_CONTACTS" />
</manifest>
```

其中`<uses-feature>`可以给应用声明功能，`<uses-permission>`可以给应用声明权限。

## （三）解析信息类

解析信息类`ResolveInfo`（对intent-filter）是`PackageItemInfo`（间接）子类。一般通过一个intent匹配来获取，通常用于判断是否存在匹配intent的Activity。

```java
PackageManager pm = getActivity().getPackageManager();
List<ResolveInfo> resolvesInfo = pm.queryIntentActivities(aIntent, 0);    // 获取能匹配aIntent的所有Activity的解析信息
```

在PacakgeManager返回的ResolveInfo对象中，可以获得Activity的标签信息，如`loadLabel()`通常是包名、`loadIcon()`是图标；`ResolveInfo.activityInfo`得到它所对应的Activity信息，可获得其类所在的包名、类名等，如`activityInfo.applicationInfo.packageName`、`activityInfo.name`等一些其他元数据。

利用集合工具类Collections的排序算法根据label标签对ResolveInfo对象排序：

```java
Collections.sort(resolvesInfo, new Comparator<ResolveInfo>() {
    public int compare(ResolveInfo a, ResolveInfo b) {
        PackageManager pm = getActivity().getPackageManager();
        return String.CASE_INSENSITIVE_ORDER.campare(a.loadLabel(pm).toString(), b.loadLabel(pm).toString());
    }
});
```

可以通过ResolveInfo构造显示Intent以启动其他应用Activity。如：

```java
ActivityInfo activityInfo = mResolveInfo.activityInfo;
Intent intent = new Intent(Intent.ACTION_MAIN)
    .setClassName(activityInfo.applicationInfo.packageName, activityInfo.name)
    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
startActivity(intent);
```

# 二、视图布局

## （一）Fragment

### 1. 创建一个Fragment

fragment类似于activity，并由activity托管，它的生命周期：`onCreate()`、`onCreateView()`、`onAttach()`、`onStart()`、`onResume()`、`onPause()`、`onStop()`、`onDestoryView()`、`onDestory()`、`onDetach()`。

fragment需要一个容器，容器视图的xml文件（通常是activity或父fragment），它们通常使用`FrameLayout`来作为容器，FrameLayout组件和其他普通组件一样。

```xml
<!-- activity_main.xml -->
    <FrameLayout
        android:id="@+id/id_ma_frame_layout"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        app:layout_constraintBottom_toTopOf="@+id/id_ma_button"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toTopOf="parent" />
```

fragment子类需要继承自Fragment类（支持库中的），覆盖父类的方法：

```java
// MainFragment.java
public class MainFragment extends Fragment {
    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // do somethings.
    }
    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_main, container, false);
        // else layout instancing or operation.
        return view;
    }
}
```

在activity中创建`FragmentManager`以管理fragment队列：

```java
// MainActivity.java @ onCreate()
        FragmentManager fragmentManager = getSupportFragmentManager();
        Fragment fragment = fragmentManager.findFragmentById(R.id.id_ma_frame_layout);
        if (fragment == null) {
            fragment = new MainFragment();
            fragmentManager.beginTransaction()
                    .add(R.id.id_ma_frame_layout, fragment)
                    .commit();
        }
```

### 2. 不同activity托管的Fragment之间的信息传输

从Fragment里启动另一个Activity的方法与从activity里启动类似，可以通过Intent传递数据；而这个新Activity里面同样可以托管Fragment，这个新Fragment如何从新Activity里面取得数据，有两种方法。

第一种方法是直接在Fragment中使用`getActivity().getIntent().getXXX();`方法。

而推荐另一种方法是使用fragment argument，通常给Fragment声明个newInstance的方法，用参数接收所需要的数据，创建自身并存储参数，返回所创建的Fragment自身。如：

```java
// MainFragment.java
public class MainFragment extends Fragment {
    /* xxx */
    public static Fragment newInstance(String str) {
        Bundle args = new Bundle();
        args.putString("key", str);
        Fragment self = new MainFragment();
        self.setArguments(args);
        return self;
    }
}
```

在Activity中启动Fragment时就可以将它需要传输的数据，作为newInstance方法是参数传给新Fragment，而新Fragment也可以在自身中利用所传输的数据。

```java
// MainActivity.java
fragment = MainFragment.newInstance("data");
// MainFragment.java
String str = getArguments().getString("key");
```

不过Fragment没有setResult方法，新Fragment想要返回数据，必须通过所托管它的Activity的setResult方法，即`getActivity().setResult(key, value);`。

### 3. 同一个activity托管的fragment之间的信息传输

在同一个activity所托管的两个Fragment之间数据传输。若是原fragment（Old Fragment，OF）向新fragment（New Fragment，NF）里传入数据，可以使用上述argument方法。

若从新fragment里面向原fragment传回结果，可调用OF的onActivityResult()。先在OF里获得要启动的NF实例，然后为其设置目标Fragment，即将OF设置为NF的目标fragment，需要调用它的`setTargetFragment()`方法。

```java
// OldFragment.java
public static final int Request_Code = 0x1;    // 请求代码

NewFragment nf = NewFragment.newInstance("data");
nf.setTargetFragment(OF.this, Request_Code);
nf.show(fragmentManager, 键字符串);    // 显示新fragment
```

之后再覆盖OF的`onActivityResult()`方法，以方便在NF中需要传回结果的时候回调。

```java
// OldFragment.java
@Override
public void onActivityResult(int requestCode, int resultCode, Intent data) {
    if (resultCode != Activity.RESULT_OK) return;
    if (requestCode == Request_Code) {
        数据 = data.getSerializableExtra("key");
    } else { /* xxx */ }
}
```

最后如果需要在NF中向OF中传回结果时，就可将数据包装好，调用OF的onActivityResult方法就可以了。

```java
// NewFragment.java
public static final int Result_Code = 0x02;

Intent data = new Intent();
data.putExtra("key", "value");
getTargetFragment().onActivityResult(getTargetRequest(), Result_Code, data);
```

## （二）RecyclerView列表

支持库`recyclerview`，添加如下：在工具栏中选择`File > Project Structure`，在新打开的Project Structure窗口中选择左侧列表中的`Dependencies`，右边Modules一栏中选择`app`，在最右边的Declared Dependencies中点击加号`+`，选择`1 Library Dependency`，搜索栏填入`androidx.recyclerview:recyclerview:Google`，点击Search搜索，选择所需的结果，点击OK。待下载重编译后则导入成功。然后就可以使用了，在xml中如`androidx.recyclerview.widget.RecyclerView`。

然后就可以在java类中使用它了。设置RecyclerView的xml视图布局id，关联xml到代码：

```java
// MainFragment.java @ onCreateView()
mRecyclerView = view.findViewById(R.id.id_mf_recycler_view);    // 在之前设置的 fragment_main.xml 文件中定义
mRecyclerView.setLayoutManager(new LinearLayoutManager(getActivity()));
```

RecyclerView上列表的每一项是一个`ViewHolder`（它引用着itemView），客户一般使用它的子类，通常定义为Fragment的内部类。

```xml
<!-- holder_item.xml -->
<LinearLayout
    android:id="@+id/id_ih_linear_layout" >
    <Button
        android:id="@+id/id_ih_button" />
</LinearLayout>
```

绑定列表项，在ItemHolder中实现列表项每个项的视图布局代码，并提供一个bind方法接收由adapter传给的数据对象，并根据数据设置好视图显示，这个bind方法在Adapter中调用。通常而言，将每一项的点击事件放在Holder中，先将Holder实现View.OnClickListener接口，在构造方法中设置监听器，然后实现监听方法。

```java
// MainFragment.java @ MainFragment
    private class ItemHolder extends RecyclerView.ViewHolder implements View.OnClickListener {
        private Button mItemButton;
        public ItemHolder(LayoutInflater inflater, ViewGroup parent) {
            super(inflater.inflate(R.layout.holder_item, parent, false));
            itemView.setOnClickListener(this);
        }
        public void bind(String name) {
            mItemButton = itemView.findViewById(R.id.id_ih_button);
            mItemButton.setText(name);
        }
        @Override
        public void onClick(View v) {
            // handle the click event.
        }
    }
```

Adapter是一个控制器对象，它从模型层获取数据（存储数据），提供给RecyclerView，通常使用Adapter的子类，定义为Fragment的子类。Adapter有一些数据更新方法，如`notifyDataSetChanged()`。

```java
// MainFragment.java @ MainFragment
    private class RecyclerAdapter extends RecyclerView.Adapter {
        private List<String> mNameList;
        public RecyclerAdapter(List<String> list) {
            mNameList = list;
        }

        @NonNull
        @Override
        public RecyclerView.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            LayoutInflater layoutInflater = LayoutInflater.from(getActivity());
            return new ItemHolder(layoutInflater, parent);
        }
        @Override
        public void onBindViewHolder(@NonNull RecyclerView.ViewHolder holder, int position) {
            if (holder instanceof ItemHolder) {
                ((ItemHolder)holder).bind(mNameList.get(position));
            }
        }
        @Override
        public int getItemCount() {
            return mNameList.size();
        }
    }
```

在它的外部类（此处为MainFragment）中添加一个RecyclerAdapter实例，然后为RecyclerView设置它的Adapter，如：`mRecyclerView.setAdapter(mAdapter);`。

### 1. 列表项的拖动和侧滑删除

用`ItemTouchHelper`可以实现列表项的拖动和侧滑删除。首先要在持有RecyclerView的activity或fragment中添加ItemTouchHelper成员，并在自定义的Adapter子类中实现两个辅助方法。此处仍使用MainFragment例子。

```java
// MainFragment.java
public class MainFragment {
    private ItemTouchHelper mItemTouchHelper;
    /* xxx */
    private class RecyclerAdapter extends RecyclerView.Adapter {
        /* xxx */
        // 两项交换
        public void onItemMove(int srcPosition, int destPosition) {
            Collections.swap(mNameList, srcPosition, destPosition);
            notifyItemMoved(srcPosition, destPosition);
        }
        // 删除一项
        public void onItemRemove(int position) {
            mNameList.remove(position);
            notifyItemRemoved(position);
        }
    }
}
```

创建ItemTouchHelper的时候需要一个`ItemTouchHelper.Callback`类型的对象作为构造参数，用来指定ItemTouchHelper处理对应事件时所调用的回调方法。使用时一般自定义一个ItemTouchHelper.Callback子类实现，该子类通常作为持有RecyclerView的activity或fragment的内部类。

```java
// MainFragment.java
public class MainFragment {
   /* xxx */
    private class myCallback extends ItemTouchHelper.Callback {
        @Override
        public int getMovementFlags(RecyclerVIew recyclerView, RecyclerView.ViewHolder viewHolder) {
            int dragFlags = ItemTouchHelper.UP | ItemTouchHelper.DOWN;
            int swipeFlags = ItemTouchHelper.LEFT;
            return makeMovementFlags(dragFlags, swipeFlags);
        }
        @Override
        public boolean onMove(RecyclerVIew recyclerView, RecyclerView.ViewHolder viewHolder, RecyclerView.ViewHolder target) {
            mAdapter.onItemMove(viewHolder.getAdapterPosition(), target.getAdapterPosition());
            return true;
        }
        @Override
        public void onSwipe(RecyclerView.ViewHolder viewHolder, int direction) {
            mAdapter.omItemRemove(viewHolder.getAdapterPosition());
        }
    }
}
```

最后在持有RecyclerView的activity或fragment中，为创建ItemTouchHelper实例，并将它绑定到RecyclerView的实例。

```java
// MainFragment.java
public class MainFragment {
    @Override
    public View onCreateView(/* xxx */) {
        /* xxx */
        mItemTouchHelper = new ItemTouchHelper(new myCallback());
        mItemTouchHelper.attchToRecyclerView(mRecyclerView);
    }
}
```

### 2. 为RecyclerView添加滚动监听器

自定义ScrollListener子类以客制化覆盖需要的回调方法，然后调用RecyclerView的`addOnScrollListener()`方法将其添加给RecyclerView对象。一般要使用RecyclerView的布局控制对象，如LinearLayoutManager对象。

```java
public class ThirdActivity extends AppCompatActivity {
    /* xxx */
    private RecyclerView mRecyclerView;
    private LinearLayoutManager mLinearLayoutManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        mRecyclerView = findViewById(R.id.id_ta_recycler_view);
        mLinearLayoutManager = new LinearLayoutManager(this);
        mRecyclerView.setLayoutManager(mLinearLayoutManager);

        mRecyclerView.addOnScrollListener(new RecyclerView.OnScrollListener() {
            @Override    // 该回调方法在滚动完成后调用
            public void onScrolled(@NonNull RecyclerView recyclerView, int dx, int dy) {
                super.onScrolled(recyclerView, dx, dy);
                // 判断是否到达底部，在滚动完成后显示
                if (dy >= 0) {
                    int first = mLinearLayoutManager.findFirstVisibleItemPosition();
                    int count = mLinearLayoutManager.getChildCount();
                    int sum = mLinearLayoutManager.getItemCount();
                    if (first + count >= sum) {
                        Toast.makeText(ThirdActivity.this, "To Bottom", Toast.LENGTH_SHORT).show();
                    }
                }
            }

            @Override
            public void onScrollStateChanged(@NonNull RecyclerView recyclerView, int newState) {
                super.onScrollStateChanged(recyclerView, newState);
                // 该方法也可以判断是否到达底部
                if (!recyclerView.canScrollVertically(1)) {
                    Toast.makeText(ThirdActivity.this, "To Bottom", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }
}
```

- `canScrollVertically()`和`canScrollHorizontally()`都是View的方法，参数为左和上为-1，右和下为1，表示相机的移动方向。

### 3. 下拉刷新

实现下拉刷新的方法有很多，此处仅记录一个不太实用的方法，主要是提供参考。值得注意的是，对于RecyclerView来说，所谓的刷新仅是显示正在刷新的UI界面，具体的业务操作请不要放在主线程。

添加一个与普通的holder_item对应的，在刷新时显示在最上方或最下方的视图文件，如l`ist_item_bottom.xm`l布局文件。然后为其编写一个模型类，如`BottomVIewHoder extends ViewHoder`，可以添加方法`setVisible()`控制其是否显示。

重构相应的Adapter子类，添加相应的ViewHolder的Type以用来标识，并根据不同的标识，修改相应的行为即可。

```java
// MainFragment.java @ MainFragment
    private class RecyclerAdapter extends RecyclerView.Adapter {
        private static final int NORMAL_TYPE = 0;    // 普通ViewHolder
        private static final int BOTTOM_TYPE = 1;    // 最后一个ViewHolder
        /* xxx */
        
        @Override
        public int getItemViewType(int position) {
            // 如果是最后一个，则返回相应的Type
            if (position == getItemCount() - 1) return BOTTOM_TYPE;
            else return NORMAL_TYPE;
        }
        
        @NonNull
        @Override    // 根据不同的Type返回不同的ViewHolder
        public RecyclerView.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            if (viewType == NORMAL_TYPE) return xxx;
            else if (viewType == BOTTOM_TYPE) return xxx;
        }
        @Override
        public void onBindViewHolder(@NonNull RecyclerView.ViewHolder holder, int position) {
            if (holder instanceof NormalItemHolder) { }
            else if (holder instanceof BottomViewHolder) {
                holder.setVisible(View.VISIVLE);    // 显示
                /* xxx */
                new Handler.postDelayed(new Runnable() {
                   public void run() {
                       holder.setVisible(View.GONE);
                   } 
                }, 1000);    // 1s后设置消失
            }
        }
        @Override
        public int getItemCount() {
            return mNameList.size() + 1;    // 注意加1
        }
    }
```

### 4. 动态指定Grid布局每个项的大小

以下例子互相独立，相不干扰。

- 根据一个项确定的位置，来确定所占的列数。

```java
GridLayoutManager gridLayoutManager = new GridLayoutManager(this, 3);
gridLayoutManager.setSpanSizeLookup(new GridLayoutManager.SpanSizeLookup() {
    @Override
    public int getSpanSize(int position) {
        return position % 3;    // 根据position返回所占列数
    }
});
mRecyclerView.setLayoutManager(gridLayoutManager);
```

- 根据屏幕宽度（组件）宽度，动态更改GridLayoutManager的列数。

```java
ViewTreeObserver recyclerViewObserver = mRecyclerView.getViewTreeObserver();
recyclerViewObserver.addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
    @Override
    public void onGlobalLayout() {
        float scale = getResources().getDisplayMetrics().density;
        int widthPx = mRecyclerView.getWidth();        // 返回的单位是px
        int widthDp = (int)(widthPx / scale + 0.5f);    // 转化成dp，因为视图指定的宽度是dp，当然也可以更改视图的单位
        int spanCount = Math.round(widthDp / 120);  // 一行多少个
        mRecyclerView.setLayoutManager(new GridLayoutManager(this, spanCount));        // 设置网格布局管理
        mRecyclerView.getViewTreeObserver().removeOnGlogalLayoutListener(this);        // 设置完成后，移除这个全局监听
    }
});
```

## （三）ViewPager支持左右滑动

使用ViewPager支持左右滑动，与RecyclerView类似，不过它的“列表”项是一个Fragment；需要一个适配器`PagerAdapter`或其子类`FragmentStatePagerAdapter`、`FragmentPagerAdapter`，子类简化了父类，需要传入一个FragmentManager实例。

适配器主要实现的方法有：`@Override public Fragment getItem(int position);`、`@Override public int getCount();`等。

ViewPager有方法：`setAdapter();`、`setCurrentItem(int position);`等。

## （四）DialogFragment对话框类

一个简单实现如下：

```java
// SecondDialog.java
public class SecondDialog extends DialogFragment {
    @NonNull
    @Override
    public Dialog onCreateDialog(@Nullable Bundle savedInstanceState) {
        View view = LayoutInflater.from(getActivity()).inflate(R.layout.fragment_second_dialog, null);
        return new AlertDialog.Builder(getActivity())
                .setView(view)    // 这个view是显示在dialog中的view，可以没有
                .setTitle("AlertDialog")
                .setPositiveButton(android.R.string.ok, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        Toast.makeText(getActivity(), "Hello", Toast.LENGTH_SHORT).show();
                    }
                })
                .create();
    }
}
```

在需要对话框的地方，如在Activity或Fragment的点击事件中，可以显示DialogFragment，如：

```java
// MainActivity.java @ Button.onClick()
new SecondDialog().show(getSupportFragmentManager(), "Dialog");
```

## （五）工具栏菜单

使用工具菜单栏，需要在res/menu目录下添加工具栏的xml布局文件。如：

```xml
<!-- menu_main.xml -->
<menu xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto">

    <item android:id="@+id/id_mm_add"
        android:icon="@drawable/ic_action_add"
        android:title="@string/text_add"
        app:showAsAction="ifRoom" />
</menu>
```

其中一个item即为工具栏中的一个项。其中icon为图标，它可以使用Android Asset Studio来创建，比如使用Image Asset；其中app:showAsAction为图标的显示方式，ifRoom表示如果空间足够就显示在工具栏，否则就隐藏在列表中。

要使用这个xml文件所定义的工具栏样式，可以在使用它的activity或fragment中，覆盖onCreateOptionsMenu方法。

```java
// MainActivity.java
@Override
public boolean onCreateOptionsMenu(Menu menu) {
    getMenuInflater().inflate(R.menu.menu_main, menu);
    return true;
}

// MainFragment.java
@Override
public void onCreateOptionsMenu(Menu menu, MenuInflater inflater) {
    super.onCreateOptionsMenu(menu, inflater);
    inflater.inflate(R.menu.menu_main, menu);    
}
```

注：由于在Fragment中是由FragmentManager来调用的，要先在Fragment让FragmentManager知道fragment使用了工具栏，推荐在Fragment的onCreate方法中调用`setHasOptionsMenu(ture);`方法。

覆盖onOptionsItemSelected方法，以客制化工具栏中的项被点击时的响应。

```java
// MainActivity.java 或 MainFragment.java    
    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        switch (item.getItemId()) {
            case R.id.id_mm_add:
                Toast.makeText(Context, "Add", Toast.LENGTH_SHORT).show();
                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }
```

有一些操纵工具栏和它的项的方法。

```java
// MainFragmet.java
((AppCompatActivity)getActivity()).getSupportActionBar().setSubtitle("subTitle");    // 设置工具栏的子标题

@Override
public boolean onOptionsItemSelected(MenuItem item) {
    item.setTitle("Hello");    // 设置item项的标题
    /* xxx */
}

// 销毁工具栏，这样Android会重建工具栏以达到刷新工具栏的目的，来更新其显示的信息（如文字等）
getActivity().invalidateOptionsMenu();
```

## （六）双版面主从布局

双版面主从布局适用于平板设备，使用资源别名文件引用不同的布局文件。在res/value目录下创建`refs.xml`文件，该文件中的item用来引用正常手机模式下的MainActivity布局文件。如下：

```xml
<!-- refs.xml -->
<resource>
    <item name="activity_main_ref" type="layout">@layout/activity_main</item>
</resource>
```

再在res/value目录下创建`refs.xml`文件，只是在创建时选中`smallest screen width`设置为`600dp`，该文件中的item引用平板设备模式下的MainActivity布局文件，其中activity_main_twopane是专门为平板设备创建的xml布局文件，有两个或所需个数的FrameLayout。如下：

```xml
<!-- refs-600dp.xml -->
<resource>
    <item name="activity_main_ref" type="layout">@layout/activity_main_twopane</item>
</resource>
```

则在使用时，则可以直接使用资源别名，如：`setContentView(R.layout.activity_main_ref);`，系统会根据设备的资源配置，自动选择合适的布局文件。

对于平板模式所显示在同一屏幕（activity）上的不同的Fragment，可以考虑在所需Fragment中自定义所需回调函数接口来实现数据通讯，且又保持Fragment的可重用性。

## （七）主题样式

在AndroidManifest文件中的application元素中，修改android:theme属性来使用自定义主题。

定义颜色，可以在`res/values/colors.xml`文件中，定义一个color元素（颜色），如下：

```xml
<!-- colors.xml -->
<resources>
    <color name="dark_blue">#005A8A</color>
</resources>
```

定义样式，可以在`res/values/styles.xml`文件中，定义一个style元素（样式），如下：

```xml
<!-- styles.xml -->
<resources>
    <style name="MyButton">
        <item name="android:background">@color/dark_blue</item>
    </style>
</resources>
```

- 样式也可以继承父样式、覆盖、添加属性。根据继承范围不同：在同一个库中可以使用`<style name="父样式.Strong>"`属性来实现继承；在跨库继承时，一般要用`<style parent="父样式">`属性来实现继承。

定义主题，可以在`res/values/styles.xml`文件中，定义一个style元素（主题），主题不过是一个更为复杂的style元素。主题元素通常过于复杂，一般通过继承父主题，并覆盖所要自定义的父主题属性，来实现自定义主题。如下：

```xml
<!-- styles.xml -->
    <style name="MyAppTheme" parent="Theme.AppCompat">
        <item name="colorPrimary">@color/red</item>
        <item name="colorPrimaryDark">@color/dark_red</item>
        <item name="colorAccent">@color/dark_blue</item>
        <!-- else -->
    </style>
```

- 通过查询一个style元素的继承树，找到所要自定义的属性，来实现客制化。如窗口背景`<item name="android:windowBackground">@color/green</item>`。

还可以在主题中设置组件样式。需要注意的是，若是继承的父主题，若要覆盖父主题的某一样式，需要将自定义的样式继承于这一个父样式。如下：

```xml
<!-- styles.xml -->
    <style name="MyButton" parent="Widget.AppCompat.Button">
        <item name="android:background">?attr/colorAccent</item>
    </style>

    <style name="MyAppTheme" parent="Theme.AppCompat">
        <item name="buttonStyle">@style/MyButton</item>
    </style>
```

- 其中MyButton使用了`?attr`属性，它表示这个值是可以变化的，它会根据它之后所引用的名称如colorAccent，从当前主题里获取值，从而实现根据主题的不同，而实现的值不同。可以使用系统自带的attr，也可以在`res/values/attrs.xml`中自定义属性。

## （八）Shape Drawable

在定义单个组件的外层嵌套一个Fragment，封装到frame布局里，这样在拉伸时，frame布局会被拉伸而其内部组件如按钮或其他组件不会。

使用Shape Drawable实现按钮在按下或松开时显示的视图不同。基本思路是：使用一个shape元素描述一个状态下的样式，drawable；用layer-list元素可以将许多shape组合成一个整体，layer list drawable；用一个selector元素引用在不同状态的下的shape，state list drawable。

一个示例如下。在res/drawable目录下，创建一个xml文件，表示平常状态，如my_button_normal.xml：

```xml
<!-- my_button_normal.xml -->
<shape xmlns:android="http://schemas.android.com/apk/res/android"
    android:shape="oval" >
    <solid android:color="@color/green" />
</shape>
```

在res/drawable目录下，创建一个xml文件表示按下状态，使用layer list drawable以将多个xml drawable合而为一，如my_button_pressed：

```xml
<!-- my_button_pressed.xml -->
<layer-list xmlns:android="http://schemas.android.com/apk/res/android">
    <item>
        <shape android:shape="oval">
            <solid android:color="@color/red"/>
        </shape>
    </item>
    <item>
        <shape android:shape="oval">
            <stroke android:width="4dp"
                android:color="@color/dark_red" />
        </shape>
    </item>
</layer-list>
```

在res/drawable目录下，创建一个xml文件表示所定义的组件，它用state list drawable将不同状态下的视图结合起来，以供程序引用，如my_button.xml：

```xml
<!-- my_button.xml -->
<selector xmlns:android="http://schemas.android.com/apk/res/android">
    <item android:drawable="@drawable/my_button_pressed"
        android:state_pressed="true" />
    <item android:drawable="@drawable/my_button_normal" />
</selector>
```

最后在样式中就可以使用，如在res/values/styles.xml中：

```xml
<!-- styles.xml -->   
    <style name="MyButton" parent="Widget.AppCompat.Button">
        <item name="android:background">@drawable/my_button</item>
    </style>
```

使用9-patch图像可以用来处理拉伸问题，从而在一定程度上避免拉伸失真问题。直接将文件名改为以`.9.png`为后缀（Refactor > Rename），如果有提示，保存默认继续即可。双击打开内置的9-patch编辑器，可以勾选`show patches`（拉伸范围为顶部和左部的黑条），`show content`（拉伸范围为底部和右部的黑条）。用鼠标拖动来编辑。

## （九）使用SearchView搜索框

组件`SearchView`是搜索框，可以将它嵌入到工具栏中，示例过程如下。

在res/menu的工具栏的xml文件中添加搜索框的item，如：

```xml
<!-- menu_fourth_activity.xml -->
<menu>
    <item android:id="@+id/id_fam_search_view"
        android:title="Search"
        app:actionViewClass="androidx.appcompat.widget.SearchView"
        app:showAsAction="ifRoom" />
    <item android:id="@+id/id_fam_clear_search"
        android:title="Clear Search"
        app:showAsAction="never" />
</menu>
```

然后再在Activity或者Fragment中实例化它，实现控制代码。注意实例化时使用的是androidx库的还是android库的。

```java
// FourthActivity.java
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_fourth_activity, menu);
        SearchView searchView = (SearchView) menu.findItem(R.id.id_fam_search_view).getActionView();
        searchView.setOnQueryTextListener(new SearchView.OnQueryTextListener() {
            @Override
            public boolean onQueryTextSubmit(String query) {
                // 存储搜索框信息，如到Shared Preferences中
                // updateItems();
                searchView.onActionViewCollapsed(); // 用户提交结果后，应撤销键盘
                return true;
            }
            @Override
            public boolean onQueryTextChange(String newText) { return false; }

        });
        searchView.setOnSearchClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // 再次点击时，恢复上次搜索框内容，如从Shared Preferences中
                searchView.setQuery("query_things", false);
            }
        });
        return true;
    }
```

同时设置清除搜索框内容的方法：

```java
    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        switch (item.getItemId()) {
            case R.id.id_fam_clear_search:
                // 清楚存储的搜索框信息，如Shared Preferences中信息置null
                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }
```

## （十）控制音频/视频播放进度条

以音频为例，采用Android原生库，需要用到`SeekBar`进度条，`MediaPlayer`库（以毫秒处理），`Handler`处理线程。一个例子如下。

```java
// MainActivity.java
public class MainActivity extends AppCompatActivity {
    private String mMusicPath = "/_ForAndroidTest/Mass.mp3";
    private SeekBar mSeekBar;
    private Button mButton;

    private MediaPlayer mMediaPlayer;
    private Handler mHandler;
    private Runnable mUpdateThread;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        PermissionCenter.requestPermissionHelper(this);    // 申请外部存储权限

        mMusicPath =  Environment.getExternalStorageDirectory().getAbsolutePath() + mMusicPath;
        mSeekBar = findViewById(R.id.id_ma_seekBar);
        mButton = findViewById(R.id.id_ma_button);

        // 创建MediaPlayer实例和初始化，也可用其他方式，如静态方法MediaPlayer.create()
        mMediaPlayer = new MediaPlayer();
        try {
            mMediaPlayer.setDataSource(mMusicPath);
            mMediaPlayer.prepare();
        } catch (IOException e) {
            e.printStackTrace();
        }

        mSeekBar.setMax(mMediaPlayer.getDuration());    // 毫秒数

        // 处理进度条和音频播放之间的同步逻辑
        mHandler = new Handler();
        mUpdateThread = new Runnable() {
            @Override
            public void run() {
                mSeekBar.setProgress(mMediaPlayer.getCurrentPosition());
                mHandler.postDelayed(mUpdateThread, 100);
            }
        };

        // 播放或者暂停监听
        mButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (mMediaPlayer.isPlaying()) {
                    mMediaPlayer.pause();
                    mHandler.removeCallbacks(mUpdateThread);
                } else {
                    mMediaPlayer.start();
                    mHandler.post(mUpdateThread);
                }
            }
        });

        // 为进度条设置监听器
        mSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (fromUser) {
                    mMediaPlayer.seekTo(progress);
                }
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) { }
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) { }
        });
    }
}
```

# 三、数据相关

## （一）关联到SQLite数据库

一般将与数据库相关的业务单独放在一个包中，例如`package com.example.mytest.database;`。

### 1. 数据模式

对于一个特定需求的数据模式Schema，一般将其封装在一个特定的类中，一般在schema类中使用静态内部类封装数据库的信息，如表名、数据项名等。

```java
// BookDatabaseSchema.java
public class BookDatabaseSchema {
    public static final class Table {
        public static final String NAME = "books";
        public static final class Cols {
            public static final String UUID =       "uuid";
            public static final String NAME =       "name";
            public static final String AUTHOR =     "author";
            public static final String YEAR =       "year";
            public static final String PRICE =      "price";
        }
    }
}
```

### 2. 打开数据库

正式创建数据库，需要先打开数据库，Android提供了打开辅助类`SQLiteOpenHelper`，使用时一般定义一个它的子类，在它里面一般有代表数据库版本号和名称的静态常量，和打开数据库的操作逻辑onCreate。它会辅助实习判断数据库是否存在（如果不存在就创建onCreate并初始化），版本是否对应等逻辑。

```java
// BookDatabaseOpenHelper.java
public class BookDatabaseOpenHelper extends SQLiteOpenHelper {

    private static final int VERSION = 1;
    private static final String Database_Name = "BookDatabase.db";

    public BookDatabaseOpenHelper(Context context) {
        super(context, Database_Name, null, VERSION);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        db.execSQL("create table " + BookDatabaseSchema.Table.NAME + "(" +
                " _id integer primary key autoincrement, " +
                BookDatabaseSchema.Table.Cols.UUID + ", " +
                BookDatabaseSchema.Table.Cols.NAME + ", " +
                BookDatabaseSchema.Table.Cols.AUTHOR + ", " +
                BookDatabaseSchema.Table.Cols.YEAR + ", " +
                BookDatabaseSchema.Table.Cols.PRICE +
                ")"
        );
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        // update the version of database
    }
}
```

其中在打开数据库类的onCreate中的方法中，数据库的execSQL方法的参数是一个字符串，注意字符串中的空格，此例当中的name、author等字符串是在Schema中定义的字段常量名。

之后就可以用一个类模拟数据库的使用了，例如：

```java
// BookDatabase.java
public class BookDatabase {
    private static BookDatabase mBase;  // 单例

    private Context mContext;
    private SQLiteDatabase mSQLiteDatabase;    // SQL数据库对象

    private BookDatabase(Context context) {
        mContext = context;
        mSQLiteDatabase = new BookDatabaseOpenHelper(mContext).getWritableDatabase();
    }
    
    public static BookDatabase get(Context context) {
        if (mBase == null) mBase = new BookDatabase(context);
        return mBase;
    }
}
```

### 3. 写入更新

负责处理数据库写入和更新操作的辅助类是`ContentValues`。它是一个键值存储类，类似于 Java的HashMap和前面用过的Bundle。不同的是，ContentValues只能用于处理SQLite数据。 想要将对象数据存入到数据库，首先要先将对象转换成ContentValues对象，ContentValues的键就是数据表字段。除了\_id是数据库自动创建，其他所有数据表字段都要编码指定。注意不要搞错，否则会导致数据插入和更新失败。

然后就可以进行写入、更新、删除数据的操作了。

```java
// BookDatabase.java
public class BookDatabase {
    /* xxx */
    private static ContentValues getContentValues(Book book) {
        ContentValues contentValues = new ContentValues();
        contentValues.put(Table.Cols.UUID, book.getUUID().toString());
        contentValues.put(Table.Cols.NAME, book.getName());
        contentValues.put(Table.Cols.AUTHOR, book.getAuthor());
        contentValues.put(Table.Cols.YEAR, book.getYear());
        contentValues.put(Table.Cols.PRICE, book.getPrice());
        return contentValues;
    }
    
    public void addBook(Book book) {
        ContentValues contentValues = getContentValues(book);
        mSQLiteDatabase.insert(Table.NAME, null, contentValues);
    }

    public void updateBookPrice(Book book) {
        String uuidString = book.getUUID().toString();
        ContentValues contentValues = getContentValues(book);    // 新的数据
        mSQLiteDatabase.update(Table.NAME, contentValues,
                Table.Cols.UUID + " = ?",    // 更新哪个字段
                new String[] { uuidString } // 值
                );    
    }
    
    public void deleteBook(Book book) {
        String uuidString = book.getUUID().toString();
        mSQLiteDatabase.delete(Table.NAME,
                Table.Cols.UUID + " = ?",
                new String[] { uuidString }
                );
    }
}
```

### 4. 查询

读取数据需要用到SQLiteDatabase.query()方法，它有好几个重载版本，其参数是一些SQL语句。这个查询方法的返回对象是一个`Cursor`对象，它相当于数据库内部的指针。直接使用Cursor不方便，Android提供了一个它的包装器`CursorWrapper`，该类继承了Cursor类的全部方法。一般使用自定义实现的它的子类。

```java
// BookCursorWrapper.java
public class BookCursorWrapper extends CursorWrapper {
    public BookCursorWrapper(Cursor cursor) {
        super(cursor);
    }

    public Book getBook() {
        String uuidString = getString(getColumnIndex(Table.Cols.UUID));
        String name = getString(getColumnIndex(Table.Cols.NAME));
        String author = getString(getColumnIndex(Table.Cols.AUTHOR));
        String year = getString(getColumnIndex(Table.Cols.YEAR));
        String price = getString(getColumnIndex(Table.Cols.PRICE));

        Book book = new Book();
        book.setUUID(UUID.fromString(uuidString));
        book.setName(name);
        book.setAuthor(author);
        book.setYear(year);
        book.setPrice(price);

        return book;
    }
}
```

接下来在模拟数据库操作逻辑的类中封装查询逻辑了，用query查询获得一个Cursor，用它生成一个包装器CursorWrapper，通过包装器的方法，读取所查询到的所有结果。注：最后不要忘记close，否则应用会抱错或崩溃。

```java
// BookDatabase.java
public class BookDatabase {
    /* xxx */
    private BookCursorWrapper queryForWrapper(String whereClause, String[] whereArgs) {
        Cursor cursor = mSQLiteDatabase.query(
                Table.NAME,
                null,
                whereClause,
                whereArgs,
                null,
                null,
                null
        );
        return new BookCursorWrapper(cursor);
    }
    
    // 返回数据库所有数据
    public List<Book> getBooks() {
        List<Book> books = new ArrayList<>();
        BookCursorWrapper cursorWrapper = queryForWrapper(null, null);
        try {
            cursorWrapper.moveToFirst();
            while (!cursorWrapper.isAfterLast()) {
                books.add(cursorWrapper.getBook());
                cursorWrapper.moveToNext();
            }
        } finally {
            cursorWrapper.close();
        }
        return books;
    }
    
    // 举例，用 uuid 查询数据
    public Book getBook(UUID id) {
        String uuidString = id.toString();
        BookCursorWrapper cursorWrapper = queryForWrapper(
                Table.Cols.UUID + " = ?",
                new String[]{ uuidString }
        );
        try {
            if (cursorWrapper.getCount() == 0) return null;
            cursorWrapper.moveToFirst();
            return cursorWrapper.getBook();
        } finally {
            cursorWrapper.close();
        }
    }
}
```

## （二）数据绑定

### 1. 启用数据绑定

如果要启动数据绑定（data binding），在app/build.gradle文件中android下，添加以下代码：`dataBinding { enable = true }`，以打开IDE的整合功能，允许使用数据绑定产生的类，并把它们整合到编译里去。

在启用数据绑定后，使用了layout的一个视图的xml布局文件，会生成对应的数据绑定类，将整个视图作为一个组件整体表示，从而不必再使用一些组件如Button来分别表示视图所含有的组件。

将一般的布局改成数据绑定布局，在最外层套一个layout标签，此处使用一个的例子：

```xml
<!-- activity_main.xml -->
<layout xmlns:android="http://schemas.android.com/apk/res/android">
    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="match_parent">
        <TextView
            android:id="@+id/id_ma_text_view"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="TextView"/>
        <Button
            android:id="@+id/id_ma_button"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="Button"/>
    </LinearLayout>
</layout>
```

之后便可以使用实例化绑定类，可以使用数据绑定辅助类`DataBindingUtil`。绑定类名为：布局文件名+Binding，它有一些方法，如`getRoot()`等，它的id资源可以直接引用。使用如：

```java
// MainActivity.java @with 布局文件 activity_main.xml
@Override
protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    ActivityMainBinding binding = DataBindingUtil.inflate(LayoutInflater.from(this), R.layout.activity_main, null, false);
    binding.idMaTextView.setText("HelloTextView");    // 用id引用资源，该例中idMaTextView是编译器根据xml里面的TextView组件的id属性值自动生成的
    binding.idMaButton.setText("HelloButton");
    setContentView(binding.getRoot());    // getRoot() 返回整个布局 View
}
```

### 2. 创建视图模型

模型视图（ViewModel）负责同步视图和模型的数据（显示）。根据上述，数据绑定类就是一个数据模型，而视图是根据布局的xml文件创建的，所以如果想要将数据模型和视图联系起来，必须通过一个中介类且利用xml来实现。这个中介类就是模型视图类。

模型视图可以根据需要完全自定义，让一个数据绑定类（即数据模型类）持有它所对应的模型视图；且让xml布局文件引用模型视图，从而可以让根据xml文件创建的视图能够接受模型视图的反馈。需要注意的是，为了使ViewModel给视图反馈，需要让自定义的视图模型类继承自`BaseObervable`类，并且用`@Bindable`注解要反馈的（绑定）属性，并显示唤醒。

针对上面给出的例子，此处创建一个数模型视图类，作为例子：

```java
// MainActivityViewModel.java
public class MainActivityViewModel extends BaseObservable {
    private String mTextViewText = "TextView_ViewModel";
    private String mButtonText = "Button_ViewModel";

    @Bindable
    public String getTextViewText() {
        return mTextViewText;
    }

    @Bindable
    public String getButtonText() {
        return mButtonText;
    }

    public void setButtonText(String buttonText) {
        mButtonText = buttonText;
        notifyChange();
    }

    public void setTextViewText(String textViewText) {
        mTextViewText = textViewText;
        notifyChange();
    }
}
```

之后可以在数据绑定布局xml文件中声明模型视图属性，作为layout元素的子`<data>`元素；并可以在xml文件中使用`@{}`来使用在模型视图类中用@Bindable注解的方法，可以使用语法糖（该例中TextView使用的普通形式，而Button使用了语法糖）。如下：

```xml
<!-- activity_main.xml -->
<layout xmlns:android="http://schemas.android.com/apk/res/android">
    <data>
        <variable
            name="viewModel"
            type="com.my.otest.MainActivityViewModel" />
    </data>
    <LinearLayout>
        <TextView
            android:text="@{viewModel.getTextViewText}"/>
        <Button
            android:text="@{viewModel.buttonText}"/>
    </LinearLayout>
</layout>
```

之后让数据绑定类（即数据模型类）持有一个数据模型类，以方便在其他位置调用数据模型类的反馈方法等。

```java
// MainActivity.java @with 布局文件 activity_main.xml
@Override
protected void onCreate(Bundle savedInstanceState) {
    /* xxx */
    binding.setViewModel(new MainActivityViewModel());
    setContentView(binding.getRoot());    // getRoot() 返回整个布局 View
}

// 在其他位置更新视图，可以调用数据模型类的一系列方法，如
binding.getViewModel().setButtonText("a button");
```

## （二）使用LruCache实现临时缓存

可以包装一个缓存类，使用它时作为Activity或Fragment的成员对象，让它和Activity或Fragment的生命周期一致，仅起到临时缓存的作用。如对Bitmap的缓存参考如下。

```java
// BitmapCache.java
public class BitmapCache {
    private LruCache<String, Bitmap> mCache;
    private float mMaxScale = 0.1f;     // 最大使用内存的比例，默认为0.2

    public BitmapCache(float scaleOfMaxMemory) {
        if (scaleOfMaxMemory <= 0 || scaleOfMaxMemory > mMaxScale) {
            scaleOfMaxMemory = mMaxScale;
        }
        int maxMemoryByte = (int) Runtime.getRuntime().maxMemory();
        int cacheByte = (int) (maxMemoryByte * scaleOfMaxMemory);
        mCache = new LruCache<String, Bitmap>(cacheByte) {
            @Override
            protected int sizeOf(String key, Bitmap value) {
                return value.getByteCount();
            }
        };
    }

    public void clearCache() {
        mCache.evictAll();    // 清除缓存
    }

    public Bitmap getBitmapFromMemoryCache(String key) {
        return mCache.get(key);
    }

    public void addBitmapToMemoryCache(String key, Bitmap bitmap) {
        if (getBitmapFromMemoryCache(key) == null) {
            mCache.put(key, bitmap);
        }
    }
}
```

## （三）使用Shared Preferences实现轻量数据存储共享

使用SharedPreferences实现轻量级数据存储分享（基于应用的沙盒文件），以便让应用的不同功能模块存储使用数据。如，数据存储可以存放用户的配置信息；数据共享可以实现从某处（如搜索框）提交信息后，在应用的其他地方获取等。一个示例如下：

```java
// UserPreferences.java
public class UserPreferences {
    private static final String PREF_SOMETHING = "something";

    public static String getSomething(Context context) {
        // 获得PER_SOMETHING上存储的内容，若不存在默认为null
        return PreferenceManager.getDefaultSharedPreferences(context)
                .getString(PREF_SOMETHING, null);
    }

    public void setSomething(Context context, String something) {
        // 在PER_SOMETHING上存储内容
        PreferenceManager.getDefaultSharedPreferences(context)
                .edit().putString(PREF_SOMETHING, something)
                .apply();
    }
}
```

## （四）资源assets

需要为应用导入或创建资源，首先要创建存放资源文件的目录，右击`app`模块，选择`New > Folder > Assets Folder`，然后根据向导便可以创建资源目录，推荐默认即可，之后在app视图下便可以发现assets文件目录。若使用的资源包含很多类别，如音乐，图片，视频等等，需要对资源进行分类，可以右击assets目录，选择`new > Directory`菜单项即可。

之后既可以将资源文件复制到资源目录下备用，可以根据需要创建一个表示资源文件的类和一个管理资源类的资源管理类等。一般而言，资源管理类持有一个Android提供的资源管理器类`AssetManager`。

需要注意的是，由于设备旋转等备选资源重配置，且音频不能用saveInstanceState保存，所以对于Fragment来说，应该保存fragment不被销毁，而只重建fragment视图，可以在Fragment的onCreate()方法中，设置fragment的属性值：`setRetainInstance(true);`。

### 1. 使用SoundPool播放音频

`SoundPool`加载音频资源到内存中，并能控制同时播放的音频文件的个数。通常作为需要音频播放功能的activity或fragment的成员，当然也可以封装一个音频播放管理类。值得注意的是，SoundPool播放音频只能持续一小段时间，故用它播放的音乐多是一小段的效果音，而且它对音频文件格式有一定要求，推荐使用mp3。


这里的例子封装了一个音频资源类，和一个音频资源管理播放类，省略了一些setter和getter方法。

```java
// Sound.java
public class Sound {
    private String mAssetPath;  // 音频在资源中的路径
    private String mName;
    private Integer mSoundId;   // SoundPool加载的音频文件都有自己的Integer型ID

    public Sound(String assetPath) {
        mAssetPath = assetPath;
        String[] components = assetPath.split(File.separator);
        String filename = components[components.length - 1];
        mName = filename.substring(0, filename.lastIndexOf("."));
    }
}
```

```java
// SoundPlayer.java
public class SoundPlayer {
    private static final String TAG = "SoundPlayer";
    private static final String SOUNDS_FOLDER = "sounds";   // 目录/assets/sounds
    private AssetManager mAssetManager;
    private List<Sound> mSounds = new ArrayList<>();

    private static final int Max_Sounds = 5;
    private SoundPool mSoundPool;

    public SoundPlayer(Context context) {
        mAssetManager = context.getAssets();
        mSoundPool = new SoundPool(Max_Sounds, AudioManager.STREAM_MUSIC, 0);   // 其中STREAM_MUSIC是音乐和游戏常用的音乐控制常量
        loadSounds();
    }

    private void loadSounds() {
        String[] soundNames;
        try {
            soundNames = mAssetManager.list(SOUNDS_FOLDER); // 用资源管理器读取列表
            Log.i(TAG, "Found " + soundNames.length + " sounds");
        } catch (IOException e) {
            Log.e(TAG, "Could not list assets.", e);
            return;
        }
        for (String filename : soundNames) {
            try {
                String assetPath = SOUNDS_FOLDER + File.separator + filename;
                Sound sound = new Sound(assetPath);
                load(sound);    // 载入音频文件
                mSounds.add(sound);
            } catch (IOException e) {
                Log.e(TAG, "Could not load sound " + filename, e);
            }
        }
    }

    private void load(Sound sound) throws IOException {
        AssetFileDescriptor afd = mAssetManager.openFd(sound.getAssetPath());   // 用AssetManager来打开（载入）asset中的文件
        int soundId = mSoundPool.load(afd, 1);
        sound.setSoundId(soundId);
    }

    // 播放音频
    public void play(Sound sound) {
        Integer soundId = sound.getSoundId();
        if (soundId == null) return;
        // 其参数从左到右依次为：音频ID、左音量、右音量、优先级（无效）、是否循环、播放速率
        mSoundPool.play(soundId, 1.0f, 1.0f, 1, 0, 1.0f);
    }
    
    // 关闭时释放资源
    public void release() {
        mSoundPool.release();
    }
}
```

一个在Activity中的使用例子如下。

```java
// MainActivity.java
public class MainActivity extends AppCompatActivity {
    /* xxx */
    private Button mButton;
    private SoundPlayer mSoundPlayer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        /* xxx */
        mSoundPlayer = new SoundPlayer(this);
        mButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Sound sound = mSoundPlayer.getSounds().get(0);
                mSoundPlayer.play(sound);
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mSoundPlayer.release();
    }
}
```

# 四、隐式Intent

## （一）概述

Intent-Filter的匹配规则此处不再赘述。使用隐式Intent来启动一个Activity或进行其他操作时，通常需要：

1. 指定要执行的操作，即对应匹配规则的action，一般可使用预设的`Intent.XXX`，作为参数传入构造函数；
2. 指定待访问数据的位置，对应匹配规则中data的uri，它也可以作为参数传入构造函数；
3. 指定所涉及操作所用的数据类型，文本还是多媒体文件等，对应匹配规则中data的mineType，可以调用Intent的方法来设置，如`intent.setType("text/plain");`。
4. 可选类别，指定何时何地如何等，对应匹配规则中的category。

注：可以在Intent中附带额外Extra的数据信息等。这个时候隐式Intent匹配的是默认的应用，如果不想使用操作系统匹配的默认的应用，可以让它每次都将可选应用列出来，以供用户选择。即activity选择器，在将intent进行start操作之前，调用`intent = Intent.createChooser(intent, "select one you want.");`。

此处例举一个发生消息的例子：

```java
Intent sendIntent = new Intent(Intent.ACTION_SEND);
sendIntent.setType("text/plain");
sendIntent.putExtra(Intent.EXTRA_TEXT, "Hello, my lover!");
sendIntent.putExtra(Intent.EXTRA_SUBJECT, "message");
sendIntent = Intent.createChooser(sendIntent, "Choose one you want.");    // 使用activity选择器
startActivity(sendIntent);    // 或其他操作
```

还可以查询联系人信息，Android操作系统有一个存储联系人的数据库。示例：

```java
// MainFragment.java
private static final int Pick_Contact_Request_Code = 0x1;

Intent pickContactIntent = new Intent(Intent.ACTION_PICK, ContactsContract.Contacts.CONTENT_URI);
startActivity(pickContactIntent, Pick_Contact_Request_Code);

// 在 onActivityResult 方法里面接收所查询到的结果
@Override
protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
    if (data == null) return;
    switch(requestCode) {
        case Pick_Contact_Request_Code:
            Uri contactUri = data.getData();
            String[] whereArgs = new String[] { ContactContract.Contacts.DISPLAY_NAME };
            Cursor cursor = getActivity().getContentResolver().query(contactUri, whereArgs, null, null, null);
            try {
                if (cursor.getCount() == 0) return;
                cursor.moveToFisrt();
                mContantName = cursor.getString(0);
            } finally {
                cursor.close();
            }
    }
}
```

需要注意的是，如果操作系统找不到匹配的activity，应用就会崩溃，可用PackageManager检查能够响应intent任务的activity。

## （二）使用Intent拍照

值得注意的是，Android API 21之后，推荐使用android.hardware.camera2。

### 1. 用intent打开相机

拍照要使用相机，可以使用对应的Intent来启动相机应用。如创建打开相机的Intent：

```java
// MainActivity.java
final Intent captureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPUTRE);
PackageManager packageManager = getPackageManager();
assert captureIntent.resolveActivity(packageManager) != null : "No activity";
```

同时相机应用拍完照片之后，需要将照片存储到本地文件中，通常是存储在请求相机的应用内部存储空间中。将应用的内部存储空间位置暴露给相机应用或其他应用，需要使用ContentProvider暴露一个应用的URI给其他应用访问使用。这里使用`FileProvider`暴露应用的内部数据目录位置Uri，并将它加入到captureIntent中。

先用一个files.xml文件（在res/xml目录下）指定想要暴露哪些位置给其他应用，当然也可以使用硬字符串指定。

```xml
<!-- files.xml -->
<paths>
    <files-path name="app_photos" path="."/>
</paths>
```

然后在AndroidManifest里面声明Provider和它的权限，下面的android:authorities是唯一确定一个Provider的字符串。

```xml
<provider android:name="androidx.core.content.FileProvider"
          android:authorities="com.example.test.fileprovider"
          android:exported="false"
          android:grantUriPermissions="true">
    <meta-data android:name="android.support.FILE_PROVIDER_PATHS"
               android:resource="@xml/files" />
</provider>
```

之后就可以生成应用的Uri信息，将它加入到intent，为所匹配到的相机应用赋予权限，然后start这个captureIntent就能够启动相机，并接收结果。

```java
// MainActivity.java
public static final int Capture_Request_Code = 0x1;    // 请求代码

File targetFile = new File(getFilesDir(), "filename");    // 要暴露的目标文件
Uri localUri = FileProvider.getUriForFile(this, "com.example.test.fileprovider", targetFile);    // 第二个参数是标识provider的android:authorities属性指定的字符串
captureIntent.putExtra(MediaStore.EXTRA_OUTPUT, localUri);

// 获得所有匹配captureIntent的可拍照的应用的activity的解析信息
List<ResolveInfo> cameraActivities = packageManager.queryIntentActivities(captureIntent, PackageManager.MATCH_DEFAULT_ONLY);
for (ResolveInfo rInfo : cameraActivities) {
    grantUriPermission(rInfo.activityInfo.packageName, localUri, Intent.FLAG_GRANT_WRITE_URI_PERMISSION);    // 给所匹配到的相机应用赋予权限
}
startActivityForResult(captureIntent, Capture_Request_Code);    // 启动这个应用
```

### 2. 显示所拍图片

可以使用ImageView组件显示图片，它有一个`setImageBitmap(Bitmap)`方法用来及将一个Bitmap对象显示到视图上。`Bitmap`对象可以使用`BitmapFactory`的静态工厂方法`decodeFile(String filepath)`来生成。不过需要注意的是，存入Bitmap的照片是不会压缩的（以至于太大48MB），故需要实现一个辅组类用来缩放图片，提供缩放和默认缩放。

```java
// PictureUtils.java
public class PictureUtils {
    public static Bitmap getScaleBitmap(String path, int destWidth, int destHeight) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(path, options);
        float srcWidth = options.outWidth;
        float srcHeight = options.outHeight;
        int inSampleSize = 1;
        if (srcHeight > destHeight || srcWidth > destWidth) {
            float heightScale = srcHeight / destHeight;
            float widthScale = srcWidth / destWidth;
            inSampleSize = Math.round(Math.max(heightScale, widthScale));
        }
        options = new BitmapFactory.Options();
        options.inSampleSize = inSampleSize;
        return BitmapFactory.decodeFile(path, options);
    }

    public static Bitmap getScaleBitmap(String path, Activity activity) {
        Point size = new Point();
        activity.getWindowManager().getDefaultDisplay().getSize(size);
        return getScaleBitmap(path, size.x, size.y);
    }
}
```

之后在Activity或Fragment的onActivityResult方法中，收回赋予相机的访问内部数据的权限，防止相机应用滥用该权限。

```java
// MainActivity.java
@Override
protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
    /* xxx */
    if (requestCode == Capture_Request_Code) {
        Uri localUri = FileProvider.getUriForFile(MainActivity.this, "com.example.test.fileprovider", targetFile);
        revokeUriPermission(localUri, Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
    }
}
```

# 五、测试

如果要进行测试，可以添加测试依赖库。

通过`File > Project Structure > Dependencies > app > + > 1 Library Dependency`，打开搜索页面。搜索`org.mockito:mockito-core:2.2.1`添加、搜索`org.hamcrest:hamcrest-junit`添加，完成对`Mockito`和`Hamcrest`库的依赖。

修改app/build.gradle文件中，dependencies下的关系，将两个库的范围从implementation修改为`testImplementation`，以避免在APK包里面捎带上无用的代码库。

打开/创建测试类，对要测试的类使用`Navigate > Test`（Ctrl+Shift+T）。若为创建，注意测试库选`JUnit4`并勾选`setUp/@Before`（预处理，相当于构造方法），其他默认。位置放在单元测试（test）下，而不是整合测试（androidTest）。

测试类中，为了实现隔离测试，除去对其他组件的依赖，可用Mockito创建虚拟对象，如：`mock(类名.class);`。测试类也可以添加一些成员，一个不成为的规定是，主测对象作为成员时其命名为mSubject。

然后就可以编写测试方法了，需要用`@Test`注解。

```java
public class SoundPlayerTest {
    private SoundPlayer mSubject;
    private SoundPool mSoundPool;

    @Before
    public void setUp() throws Exception {
        mSubject = new SoundPlayer(new AppCompatActivity());
        mSoundPool = org.mockito.Mockito.mock(SoundPool.class);
    }

    @Test
    public void testFunc1() {
        assertThat(mSubject.getSoundPool(), is(mSoundPool));    // 断言mSubject.getSoundPool()和mSoundPool属性是否有关系
    }

    @Test
    public void testFunc2() {
        Sound sound = mSubject.getSounds().get(0);
        verify(mSubject).play(sound);   // 相当于 verify(mSubject); mSubject.play(sound); ，用于检验mSubject的play()方法是否被正常调用过
    }
}
```

- 值得注意的是，通过使用 Alt + Enter 组合键，后选static import method...，然后选hamcrest-core-1.3库里的MatcherAssert.assertThat(...)方法，同样方式选择hamcrest-core-1.3库里面的Is.is()方法，和mockito.Mockito库里面的verify方法。

然后在工程目录视图下，右击app/java/package-name(test)，选择Run 'Tests' in 'package-name'，在跳出的窗口中查看是否通过测试。默认只会显示失败的测试。


# 六、基本网络连接

## （一）使用检查网络权限

在AndroidManifest中的同\<application>的层级下，添加网络访问权限，`<uses-permission android:name="android.permission.INTERNET"/>`、网络状态访问权限，`<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE">`。

通常在使用网络之前需要检查网络状态，在使用网络之前调用。有很多种方法，此处一个示例：

```java
private boolean isNetworkAvailableAndConnected() {
    ConnectivityManager cm = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
    boolean isAvailable = (cm.getActiveNetworkInfo() != null);
    boolean isConnect = isAvailable && cm.getActiveNetworkInfo().isConnected();
    return isConnect;
}
```

## （二）通过URL获得JSON对象

查阅相关网络资料，是否需要KEY、访问API格式等。然后构造特定的URL字符串，它包含要访问的相关信息，可以使用`Uri.parse().buildUpon().appendQueryParameter().build().toString();`辅组类的辅助方法构造。

然后可以通过基本HTTP等网络连接获得返回的字符串，正确的API一般会返回一个JSON对象的字符串，可以通过解析JSON对象获得所需要的信息。关于基本网络连接和解析JSON对象，请参考Java面向对象编程（第2版）。

另：将JSON对象映射成Java对象，GSON是常用的JSON对象序列和反序列化的解析类，常用方法如`toJSON()`、`fromJSON()`等，需要添加依赖`com.google.code.gson:gson:2.8.5`。

这里给出例子所需的两个方法，String UserFetcher#getUrlString(String urlSpec)、List\<UserItem> UserFetcher#fetchItems()。

## （三）使用AsyncTask后台线程

### 1. 使用AsyncTask执行网络访问任务

Android系统禁止任何主线程（即UI线程）的网络连接行为，需要在后台线程上运行代码。一般自定义一个继承于`AsyncTask`的线程工具类，常作为Activity或Fragment的内部类。一个获得相关内容列表的后台工具线程类如：

```java
// SomeActivity.java or SomeFragment.java
/* xxx */
private class FetchItemTask extends AsyncTask<Void, Void, List<UserItem>> {
    @Override
    protected List<GalleryItem> doInBackground(Void... parms) { 
        return new UserFetcher.fetchItems();
    }
}
```

- 其中AsyncTask类的三个泛型参数分别为：Params、Progress、Result。
- Params为doInBackground()方法的参数类型。
- Progress为onProgressUpdate()方法的参数类型。
- Result为doInBackground()方法的返回类型，onPostExecute()方法的参数类型。

值得注意的是，如果任务发生在Fragment中，需要额外判断。由于使用AsyncTask在后台会触发回调指令，干扰Fragment从activity接受回调指令，需要使用if语句调用`isAdded()`判断Fragment是否关联Activity，防止getActivity()方法返回空值。

当获得返回结果，更新item后，需要更新显示，为防止线程冲突，在AsyncTask中更新；而Android为防止内存踩踏，不允许从后台线程更新UI线程（即主线程），故要用AsyncTask类专门提供的`onPostExecute()`方法，它的参数为接收的doInBackground()方法的返回值。在后台线程将doInBackground()方法调用完成结束后，该方法在主线程被调用，故可以更新UI线程。

之后就可以在Activity或Fragment所需要的地方，就可以开启一个工具线程在后台运行，如：`new FetchItemTask().execute();`。

也可以显示的取消AsyncTask异步线程任务，调用它的`cancel(Boolean mayInterrupt)`方法，参数为false会将`isCancelled()`方法返回的所引用的状态设为true，AsyncTask检擦它选择尽可能提前结束；参数传入true会强制终止doInBackground()方法所在的线程。需要注意的是，一个AsyncTask对象只能execute()执行一次，再次执行会发生错误。

在遇到生命周期、设备旋转等问题时，要手动设置处理AsyncTask，相关的Loader可以有LoaderManager管理loader及其数据等工作，如使用`AsyncTaskLoader`实现子类。

### 2. 使用AsyncTask实现进度条

```java
// SomeActivity.java
private class ProgressTask extends AsyncTask<Void, Integer, Void> {
    @Override
    protected Void doInBackground(Void... voids) {
        while (mPlayer.isPlaying()) {
            Integer x = mPlayer.getCurrentPosition();
            publishProgress(x);        // 给onProgressUpdate()方法传递参数，并调用之
        }
        return null;
    }
    
    @Override
    protected void onProgressUpdate(Integer... values) {
        Integer x = values[0];
        mSeekBar.setProgress(x);    // 该onProgressUpdate()可以更新UI线程
    }
}
```

## （四）使用HandlerThread实现网络下载

实现一个自定义子类`MyDownloader`继承于`HandlerThread`，在此例中为下载图片。使用第三方库会更好，例如Picasso、Glide、Fresco等。

该线程类需要实现后台下载数据（Bytes），下载完成后转化为所需类型对象，然后调用回调方法；提供相应回调接口，可以由用户指定下载完成后的操作，回调方法中一般可以传入目标对象，用根据下载的对象设置目标对象；此外还需要提供开始方法，由用户传入URL等信息并开始下载任务；最后还需要提供clear方法，用于在结束后清楚占用资源。利用Handler、Looper、MessageQueue机制实现。

```java
// MyDownloader.java
// T 类型为下载成功后，用户指定要设置的目标对象的类型
public class MyDownloader<T> extends HandlerThread {
    private static final String TAG = "MyDownloader";
    private static final int Message_Download = 0;

    private Boolean mHasQuit = false;   // 判断线程是否退出
    private Handler mRequestHandler;    // 处理请求Handler
    private Handler mResponseHandler;   // 响应 Handler
    private ConcurrentMap<T, String> mRequestMap = new ConcurrentHashMap<>();   // 任务请求Map，<用户请求设置的目标对象, 下载所用Url>
    private DownloadCompletedCallback<T> mDownloadCompletedCallback;   // 下载完成的回调，由用户指定

    // 构造方法，传入一个UI线程创建的Handler（Handler需要Looper和它管理的消息队列，主线程默认有Looper）
    public MyDownloader(Handler responseHandler) {
        super(TAG);
        mResponseHandler = responseHandler; // 该Handler由用户线程（主线程）创建并传入，从而可以刷新UI
    }

    // 设置回调方法
    public void setDownloadCompletedCallback(DownloadCompletedCallback<T> callback) {
        mDownloadCompletedCallback = callback;
    }

    // 用户调用，传入请求的目标对象和所需URL
    public void queryThumbnail(T target, String url) {
        Log.i(TAG, "Got a URL : " + url);
        if (url == null) {
            mRequestMap.remove(target);  // 指定url为空，说明目标对象取消请求或无效，如果它已在请求Map中，移除之
        } else {
            mRequestMap.put(target, url);   // 请求有效，将目标对象和url放入请求Map
            // 包装对象，并发送消息，由处理Handler处理之（自己发送，自己接收，自己处理）
            mRequestHandler.obtainMessage(Message_Download, target).sendToTarget();
        }
    }

    @Override
    @SuppressLint("HandlerLeak")
    protected void onLooperPrepared() {
        // 在准备任务中，设置好 mRequestHandler，接收消息，处理之
        mRequestHandler = new Handler() {
            @Override
            public void handleMessage(@NonNull Message msg) {
                if (msg.what == Message_Download) {
                    T target = (T) msg.obj;     // 获取消息包装的请求处理对象
                    Log.i(TAG, "Got a request for URL : " + mRequestMap.get(target));
                    handlerRequest(target);     // 处理请求，处理所要设置对象
                }
            }
        };
    }

    // 处理请求设置的对象
    private void handlerRequest(T target) {
        try {
            String url = mRequestMap.get(target);
            if (url == null) return;
            byte[] bitmapBytes = getUrlBytes(url);  // 根据URL下载对应字节
            // 用字节解析成 Bitmap 对象
            Bitmap bitmap = BitmapFactory.decodeByteArray(bitmapBytes, 0, bitmapBytes.length);
            Log.i(TAG, "Bitmap Created");
            // 下载完成，响应，移除请求，完成目标对象的设置
            mResponseHandler.post(new Runnable() {
                @Override
                public void run() {
                    if (mHasQuit) return;  // 如果将要退出，直接返回结束
                    mRequestMap.remove(target);  // 从请求任务中移除请求对象
                    mDownloadCompletedCallback.over(target, bitmap);    // 利用用户指定回调设置目标对象
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // 线程结束时清除资源，防止挂起
    public void clearQueue() {
        mRequestHandler.removeMessages(Message_Download);
        mRequestMap.clear();
    }

    @Override
    public boolean quit() {
        mHasQuit = true;
        return super.quit();
    }

    // 获取URL所定位的源字节
    private byte[] getUrlBytes(String urlSpec) throws IOException {
        URL url = new URL(urlSpec);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        if (connection.getResponseCode() != HttpURLConnection.HTTP_OK) {
            throw new IOException(connection.getResponseMessage() + " : with " + urlSpec);
        }
        try {
            InputStream in = connection.getInputStream();
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            int bytesRead = 0;
            byte[] buffer = new byte[1024];
            while ((bytesRead = in.read(buffer)) > 0) {
                out.write(buffer, 0, bytesRead);
            }
            in.close();
            out.close();
            return out.toByteArray();
        } finally {
          connection.disconnect();
        }
    }

    // 回调类型的接口
    public interface DownloadCompletedCallback<T> {
        void over(T target, Bitmap bitmap);
    }
}
```

在Activity或Fragment中使用的一个例子如下，此处就体现出了使用泛型T的好处，它可以指定为ImangeView、ImageButton等所需的目标对象类型。

```java
// SecondActivity.java
public class SecondActivity extends AppCompatActivity {
    /* xxx */
    private String mTempUrl = "https://i0.hdslb.com/bfs/face/374156e29fce7d87ac3d169f0ab5fedbb08ff708.jpg@150w_150h.jpg";
    private MyDownloader<ImageView> mDownloader;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        /* xxx */
        
        // 在UI线程创建的Handler，默认关联getMainLooper()，不需要额外处理Looper
        Handler responseHandler = new Handler();
        mDownloader = new MyDownloader<>(responseHandler);

        mDownloader.setDownloadCompletedCallback(new MyDownloader.DownloadCompletedCallback<ImageView>() {
            @Override
            public void over(ImageView target, Bitmap bitmap) {
                Drawable drawable = new BitmapDrawable(getResources(), bitmap);
                target.setImageDrawable(drawable);
            }
        });

        // start()和getLooper()两个方法调用成功，说明线程非空
        mDownloader.start();
        mDownloader.getLooper();
        Log.i(TAG, "Background thread started.");

        // 测试
        mButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mDownloader.queryThumbnail(mImageView, mTempUrl);
            }
        });

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mDownloader.clearQueue();   // 清除资源
        mDownloader.quit(); // 退出线程
    }
}
```

## （五）网页浏览

网页浏览可以直接使用隐式Intent跳转到浏览器，如下所示。

```java
Uri uri = Uri.parse("https://www.baidu.com");
Intent intent = new Intent(Intent.ACTION_VIEW, uri);
startActivity(intent);
```

或者使用WebView视图控件，如下所示。

```java
WebView wv = findViewById(R.id.my_web_view);
wv.setWebViewClient(new WebViewClient());
wv.getSettings().setJavaScriptEnabled(true);
wv.loadUrl("https://www.baidu.com");
```

- 其中使用了WebView的setWebViewClient方法来设置一个WebViewClient，以优化WebView显示。WebViewClient是响应渲染事件的接口。
- getSettings().setJavaScriptEnabled()方法用于获得WebView的设置并启用了JavaScript脚本。
- loadUrl方法加载WebView初始时的界面显示。

对于上面的WebView控件，可以使用WebChromeClient优化显示，它是一个事件接口，用来响应那些改变浏览器中装饰元素的事件，包括JavaScript警告信息、网页图标、状态条加载以及当前页面的刷新等。要实现加载进度条的效果，可以在xml中在WebView上面添加一个水平的ProgressBar进度条，然后在代码中实现如下。

```java
ProgressBar pb = findViewById(R.id.my_progress_bar);
pb.setMax(100);
wv.setWebChromeClient(new WebChromeClient() {
    @Override
    public void onProgressChanged(WebView view, int newProgress) {
        if (newProgress == 100) {
            pb.setVisibility(View.GONE);
        } else {
            pb.setVisibility(View.VISIBLE);
            pb.setProgress(newProgress);
        }
    }

    @Override
    public void onReceivedTitle(WebView view, String title) {
        getSupportActionBar().setSubtitle(title);
    }
});
```

对于一些类来说，比如VideoView和刚才讨论的WebView，因为它们包含的数据通常太多，在因配置变更等原因而销毁时无法使用onSaveInstanceState保存，而重新加载它们有显得太慢，故Android的官方文档推荐让Activity自己处理配置变更，也就是说无需销毁再重建Activity，就能直接调整直接的视图以适应新的屏幕尺寸。

需要在相应的Manifest文件中为相应的Activity添加属性如下。

```java
<activity android:name=".MyWebActivity"
    android:configChanges="keyboardHidden|orientation|screenSize"/>
```

其中android:configChanges表示当某些配置信息发生改变时，这个Activity不会销毁重建。如果有多个配置信息，使用`|`连接。

# 七、后台服务

要使任务在后台运行，使用服务Service，它是Context的子类，能够响应Intent，服务的Intent又叫做命令（command）。Service是安卓的四大组件之一，使用时需要在AndroidManifest文件中的application元素中添加`service`声明，它也有Intent-filter匹配规则。

## （一）服务、推送

`IntentService`服务能顺序执行命令队列里的命令，通常使用IntentService自定义的子类。最基本的框架如：

```java
// PollService.java
public class PollService extends IntentService {
    private static final String TAG = "MyService";

    public PollService() {
        super(TAG);
    }

    public static Intent newIntent(Context context) {
        return new Intent(context, PollService.class);
    }

    @Override
    protected void onHandleIntent(@Nullable Intent intent) {
        /* some operations. */
    }
}
```

可以在onHandleIntent方法中使用`Notification`及其相关的类实现通知栏的通知消息，先创建一个通知频道`NotificationChannel`，再创建消息放入其中通知。

```java
// PollService.java at onHandleIntent()
@RequiresApi(api = Build.VERSION_CODES.O)
@Override
protected void onHandleIntent(@Nullable Intent intent) {
    /* some operations. */
    Intent intentActivity = MainActivity.newIntent(this);
    PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, intentActivity, 0);
    String channelId = "channel_1";
    String channelName = "ReportChannel";
    Notification notification = new NotificationCompat.Builder(this)
            .setChannelId(channelId)
            .setTicker("You have a new message")
            .setSmallIcon(android.R.drawable.ic_menu_report_image)
            .setContentTitle("New")
            .setContentText("There is a new thing")
            .setContentIntent(pendingIntent)
            .build();
    @SuppressLint("WrongConstant")
    NotificationChannel channel = new NotificationChannel(channelId, channelName, NotificationManagerCompat.IMPORTANCE_LOW);
    NotificationManagerCompat nm = NotificationManagerCompat.from(this);
    nm.createNotificationChannel(channel);
    nm.notify(1, notification);
}
```

## （二）延迟运行服务

主要使用的是`AlarmManager`来延迟运行服务。思路是先在PollService中设置推送间隔，并提供设置是否推送的静态方法；然后在用到的地方如Fragment的onCreate()中设置默认启动定时器，如下。

```java
// PollService.java at class PollService
private static final long POLL_INTERVAL_MS = TimeUnit.MINUTES.toMillis(1);

public static void setServiceAlarm(Context context, boolean isOn) {
    Intent intent = PollService.newIntent(context);
    PendingIntent pendingIntent = PendingIntent.getService(context, 0, intent, 0);
    AlarmManager am = (AlarmManager) context.getSystemService(Context.ALARM_SERVICE);
    if (isOn) {
        am.setRepeating(AlarmManager.ELAPSED_REALTIME, SystemClock.elapsedRealtime(), POLL_INTERVAL_MS, pendingIntent);
    } else {
        am.cancel(pendingIntent);
        pendingIntent.cancel();
    }
}

// FourthActivity.java at onCreate()
PollService.setServiceAlarm(this, true);
```

- 其中AlarmManager.ELAPSE_REALTIME是基准时间，表明是以SystemClock.elapseRealtime()走过的时间来启动定时器的。此外AlarmManager还有RTC、ELAPSE_REALTIME_WAKEUP等成员。

为控制定时器，需判断Alarm是否设置，由于Alarm是和PendingIntent同步设置的，可利用PendingIntent来判断，即调用它的`getService()`方法通过传入特定的Flags参数，判断得到intent是否为空，来判断Alarm是否被设置（因为对相同的intent来说，PendingIntent只能登记一个），方法如下：

```java
// PollService.java at class PollService
public static boolean isServiceAlarmOn(Context context) {
    Intent intent = PollService.newIntent(context);
    PendingIntent pendingIntent = PendingIntent.getService(context, 0, intent, PendingIntent.FLAG_NO_CREATE);
    return pendingIntent != null;
}
```

通过工具栏来控制设置定时器，代码如：

```java
case R.id.menu_item_toggle_polling:
    boolean shouldStartAlarm = !PollService.isServiceAlarmOn(getActivity());
    PollService.setServiceAlarm(getActivity(), shouldStartAlarm);
    getActivity().invalidateOptionsMenu();    // 销毁工具栏以重建刷新之
    return true;
```

在onCreateOPtionsMenu()中加上：

```java
MenuItem toggleItem = menu.findItem(R.id.menu_item_toggle_polling);
if (PollService.isServiceAlarmOn(getActivity())) {
    toggleItem.setTitle("Stop");
} else {
    toggleItem.setTitle("Start");
}
```

也可以通过代码手动启动服务，如下：

```java
Intent intent = PollService.newInstance(getActivity());
getActivity().startService(intent);
```

## （三）系统广播

系统广播broadcast intent与普通的intent不同的是，broadcast intent可以被多个叫做broadcast receiver的组件接受（一对多）。其中分为standalone receiver需在manifest配置文件中声明，在应用进程消亡时仍可被激活；还有同Fragment或Activity的生命周期绑定的dynamic receiver。

### 1. 静态注册broadcast receiver

这里继续上节的例子，如果想要在系统重启后就唤醒服务，可以使用broadcast receiver来接受`BOOT_COMPLETED`，它是系统启动完成时发送的broadcast intent。

可以先用Shared preferences存储alarm是否设置，在PollService的setServiceAlarm()方法最后存储定时器的状态，如QueryPerferences.setAlarmOn(context, isOn);。

接下来就要实现自己的广播的接受者，即自定义一个继承于`BroadcastReceiver`的类：

```java
// StartupReceiver.java
public class StartupReceiver extends BroadcastReceiver {
    private static final String TAG = "StartupReceiver";
    @Override
    public void onReceive(Context context, Intent intent) {
        Log.i(TAG, "Received broadcast intent: " + intent.getAction());
        boolean isOn = QueryPreferences.isAlarmOn(context);
        PollService.setServiceAlarm(context, isOn);
    }
}
```

在manifest配置文件中注册standalone receiver，如下。

```xml
<manifest>
    <uses-permission android:name="android.permission.RECEIVE_BOOT_COMPLETED"/>
    <application>
        <receiver android:name=".StartupReceiver">
            <intent-filter>
                <action android:name="android.intent.action.BOOT_COMPLETED"/>
            </intent-filter>
        </receiver>
    </application>
</manifest>
```

### 2. 动态登记receiver

当应用视图打开时，接受到相应的intent，设置结果代码，取消通知消息推送；当应用在后台时，动态receiver消亡，正常使用alarm推送消息。

定制化broadcast intent的发与收，自己定制Intent的Action匹配。在PollService中添加自己的action字符串：

```java
// PollService.java
public class PollService extends IntentService {
    public static final String ACTION_SHOW_NOTIFICATION = "com.bloonow.android.SHOW_NOTIFICATION";
}
```

创建自己的使用权限来限制broadcast，在配置文件manifest中添加：

```xml
<manifest>
    <permission android:name="com.bloonow.android.PRIVATE"
                android:protectionLevel="signature"/>
    <uses-permission android:name="com.bloonow.android.PRIVATE"/>
</manifest>
```

发送带有权限的broadcast，在PollService中添加：

```java
// PollService.java
public class PollService extends IntentService {
    public static final String PERM_PRIVATE = "com.bloonow.android.PRIVATE";
    @Override
    protected void onHandleIntent(@Nullable Intent intent) {
        /* xxx */
        sendBroadcast(new Intent(ACTION_SHOW_NOTIFICATION), PERM_PRIVATE);    // 发送
    }
}
```

用代码创建并登记动态receiver，取消消息推送，这里的逻辑是当某个界面显示出来时，就取消消息推送，因此自定义一个Fragment，并把它定义为抽象类，如下。

```java
// VisibleFragment.java
public abstract class VisibleFragment extends Fragment {
    private static final String TAG = "VisibleFragment";
    private BroadcastReceiver mOnShowNotification = new BroadcastReceiver {
        @Override
        public void onReceive(Context context, Intent intent) {
            Log.i(TAG, "Got a broadcast intent with action: " + intent.getAction() + "to canceling notification.");
            // 取消消息发送，可以向该Fragment的Activity发送一个返回代码，如下
            setResultCode(Activity.RESULT_CANCELED);
        }
    };
    @Override
    public void onStart() {
        super.onStart();
        IntentFilter filter = new IntentFilter(PollService.ACTION_SHOW_NOTIFICATION);
        getActivity().registerReceiver(mOnShowNotification, filter, PollService.PERM_PRIVATE, null);
    }
    @Override
    public void onStop() {
        super.onStop();
        getActivity().unregisterReceiver(mOnShowNotification);
    }
}
```

- 其中创建的IntentFilter等价于manifest配置文件中的\<intent-filter\>，实际上任何在配置文件中定义的\<intent-filter\>都可以用代码的方式定义。
- 其中PERM_PRIVATE注册了带有PERM_PRIVATE权限的动态receiver。

最后将应用视图改为继承为VisibleFragment的Fragment即可。

### 3. 视图打开时不发送推送的逻辑

使用有序broadcast收发数据，它是一对多的，但它receiver接受者不是同时接受intent的，而是有先后顺序的。通过传入一个名为result receiver的特别broadcast receiver，有序broadcast的发送者还可以从broadcast接受者那里得到返回结果。

这里要在视图打开时取消推送消息发送，通过设计broadcast的优先级，让它在视图打开的情况下，先收到广播并设置结果码，之后由PollService接受到广播的时候，根据结果码判断是否发送推送消息。

为发送有序broadcast，先在PollService中加上一个成员，写个发送消息的辅助方法，并修改onHandleIntent方法：

```java
// PollService.java
public class PollService extends IntentService {
    public static final String REQUEST_CODE = "REQUEST_CODE";
    public static final String NOTIFICATION = "NOTIFICATION";
    
    private void showBackgroundNotification(int requestCode,Notification nification) {
        Intent intent = new Intent(ACTION_SHOW_NOTIFICATION);
        intent.putExtra(REQUEST_CODE, requestCode);
        intent.putExtra(NOTIFICATION, nification);
        sendOrderedBroadcast(intent, PERM_PRIVATE, null, null,
                             Activity.RESULT_OK, null, null);
    }

    @Override
    protected void onHandleIntent(@Nullable Intent intent) {
        /* some operations. */
        Intent intentActivity = MainActivity.newIntent(this);
        PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, intentActivity, 0);
        String channelId = "channel_1";
        String channelName = "ReportChannel";
        Notification notification = new NotificationCompat.Builder(this)
            .setChannelId(channelId)
            .setTicker("You have a new message")
            .setSmallIcon(android.R.drawable.ic_menu_report_image)
            .setContentTitle("New")
            .setContentText("There is a new thing")
            .setContentIntent(pendingIntent)
            .build();
        showBackgroundNotification(0, notification);    // 发送推送的有序broadcast
    }
}
```

接下来要实现Result Receiver（用standalone receiver），并保证它是最后一个接受broadcast的，代码如下：

```java
public class NotificationReceiver extends BroadcastReceiver {
    private static final String TAG = "NotificationReceiver";
    @Override
    public void onReceive(Context c, Intent i) {
        Log.i(TAG, "received result: " + getResultCode());
        // A foreground activity cancelled the broadcast
        if (getResultCode() != Activity.RESULT_OK) {
            return;
        } 
        int requestCode = i.getIntExtra(PollService.REQUEST_CODE, 0);
        Notification n = (Notification)i.getParcelableExtra(PollService.NOTIFICATION);
        NotificationManagerCompat nm = NotificationManagerCompat.from(c);
        String channelId = "channel_1";
        String channelName = "ReportChannel";
        NotificationChannel channel = new NotificationChannel(channelId, channelName, NotificationManagerCompat.IMPORTANCE_LOW);
        nm.createNotificationChannel(channel);
        nm.notify(requestCode, n);
    }
}
```

在manifest配置文件中登记它，如下：

```xml
<manifest>
    <application>
        <receiver android:name=".NotificationReceiver"
                  android:exported="false">
            <intent-filter android:priority="-999">
                <action android:name="com.bloonow.android.SHOW_NOTIFICATION"/>
            </intent-filter>
        </receiver>
    </application>
</manifest>
```

- 其中android:exported设置为不对外部暴露。
- android:priority的优先级设成了用户可选最低的-999，以确保在PollService发送SHOW_NOTIFICATION的broadcast intent后，让VisibleFragment在有视图的情况下，跟视图绑定的receiver先于NotificationReceiver接收到intent，然后将结果码设置为REQUEST_CANCELED，确保最后NotificationReceiver就收到后，直接return，从而实现在视图打开时不推送消息。

要实现应用内的消息广播，可以使用EventBus（事件总线）、Square、Otto、Android自己的LocalBroadcastManger或用RxJava Subject和Observable来模拟事件总线。