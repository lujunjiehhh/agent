import React, { useState } from 'react';
import { Input, Button, Card, List, Typography, Space } from 'antd';
import { SendOutlined } from '@ant-design/icons';
import axios from 'axios';

const { TextArea } = Input;
const { Title, Paragraph } = Typography;

interface Reference {
  doc_id: string;
  content: string;
  relevance_score: number;
  position?: {
    start: number;
    end: number;
  };
}

interface ReasoningStep {
  step_id: string;
  description: string;
  type: string;
  details?: any;
  parent_step_id?: string;
}

interface QAResponse {
  answer: string;
  reasoning_steps: ReasoningStep[];
  references: Reference[];
}

const QAPanel: React.FC = () => {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<QAResponse | null>(null);

  const handleSubmit = async () => {
    if (!question.trim()) return;

    setLoading(true);
    try {
      const result = await axios.post('http://localhost:8000/api/qa', {
        question: question.trim(),
      });
      setResponse(result.data);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 800, margin: '0 auto' }}>
      <Card title="问答系统" style={{ marginBottom: 24 }}>
        <Space direction="vertical" style={{ width: '100%' }}>
          <TextArea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="请输入您的问题..."
            autoSize={{ minRows: 3, maxRows: 5 }}
            style={{ marginBottom: 16 }}
          />
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleSubmit}
            loading={loading}
            block
          >
            提交问题
          </Button>
        </Space>
      </Card>

      {response && (
        <Card title="回答">
          <Space direction="vertical" style={{ width: '100%' }}>
            <Title level={4}>答案</Title>
            <Paragraph>{response.answer}</Paragraph>

            <Title level={4}>推理步骤</Title>
            <List
              dataSource={response.reasoning_steps}
              renderItem={(step) => (
                <List.Item>
                  <Card size="small" style={{ width: '100%' }}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Paragraph strong>{step.description}</Paragraph>
                      {step.details && (
                        <Paragraph type="secondary">
                          {JSON.stringify(step.details, null, 2)}
                        </Paragraph>
                      )}
                    </Space>
                  </Card>
                </List.Item>
              )}
            />

            <Title level={4}>参考文档</Title>
            <List
              dataSource={response.references}
              renderItem={(ref) => (
                <List.Item>
                  <Card size="small" style={{ width: '100%' }}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Paragraph>{ref.content}</Paragraph>
                      <Paragraph type="secondary">
                        相关度: {(ref.relevance_score * 100).toFixed(2)}%
                      </Paragraph>
                    </Space>
                  </Card>
                </List.Item>
              )}
            />
          </Space>
        </Card>
      )}
    </div>
  );
};

export default QAPanel; 