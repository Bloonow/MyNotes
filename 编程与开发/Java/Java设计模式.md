# Java单例模式的泛型

## （一）重量级实现

```java
public class Singleton<T> {
    private static Map<Class<?>, Object> mInstanceMap = new HashMap<>();   // 若要并发，使用ConcurrentHashMap
    private Singleton() {}
    public static <T> T getInstance(Class<T> clazz) {
        Object obj = mInstanceMap.get(clazz);
        if (obj == null) {
            try {
                // 若要多线程工作，注意处理好并发与同步问题
                obj = clazz.newInstance();
                mInstanceMap.put(clazz, obj);
            } catch (InstantiationException | IllegalAccessException e) {
                e.printStackTrace();
            }
        }
        return (T) obj;
    }
}

// 使用的例子如下
A a = Singleton.getInstance(A.class);
```

## （二）轻量级实现

这个实现实际上由Android源代码实现的，如下所示。

```java
public abstract class Singleton<T> {
    private T mInstance;
    protected abstract T create();
    public final T get() {
        synchronized (this) {
            if (mInstance == null) {
                mInstance = create();
            }
            return mInstance;
        }
    }
}
```

其使用方式的一个例子如下。

```java
private static MyRecord getRecord() {
    return sMyRecordSingleton.get();
}

private static final Singleton<MyRecord> sMyRecordSingleton =
        new Singleton<MyRecord>() {
            @Override
            protected MyRecord create() {
                // 可自定义创建新对象的方式
                return new MyRecord();
            }
        };
```